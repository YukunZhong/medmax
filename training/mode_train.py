"""
MoDE Continual VQA Training: ScienceQA → TextVQA → VizWiz

Algorithm (from paper):
  1. Frozen backbone (Anole / ChameleonForConditionalGeneration).
  2. Frozen teacher = identical backbone copy, no adapters.
  3. Student = backbone + V-Adapter (image LoRA) + T-MoE (text MoE-LoRA)
     inserted into every MLP linear layer.
  4. Sequential task training; 1 epoch per task.
  5. Per-batch losses:
       L_CE  : cross-entropy on answer tokens  (updates T-MoE + V-Adapter)
       L_KD  : KL divergence on image-token next-token logits vs teacher
               (updates V-Adapter ONLY — T-MoE is skipped in kd_mode)
       total : L_CE  +  λ * L_KD   (via two separate .backward() calls
               before one optimizer.step())

Hyperparameters (paper defaults):
  lora_rank=8, num_experts=4, lr=1e-4, cosine+warmup=0.1,
  beta=2.0 (KD temperature), lambda_kd=0.3, 1 epoch per task.

Usage (single GPU):
  cd /data1/data/kangborui/zhongyukun/medmax/medmax
  CUDA_VISIBLE_DEVICES=0 python -m training.mode_train \\
      --ckpt /data1/data/kangborui/zhongyukun/medmax/anole_7b_hf \\
      --data_dir /data1/data/kangborui/zhongyukun/medmax/MoDE-official/data \\
      --output_dir /data1/data/kangborui/zhongyukun/medmax/output_dir_MoDE \\
      --tasks scienceqa textvqa vizwiz \\
      --task_dirs ScienceQA TextVQA VizWiz \\
      --use_caption
"""

import argparse
import copy
import itertools
import json
import os
import sys
from typing import Dict, Iterator, List, Optional

import jsonlines
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import ChameleonForConditionalGeneration, get_cosine_schedule_with_warmup

from .mode_adapters import (
    MoDEChameleonMLP,
    build_image_mask,
    compute_lce_gamma,
    compute_stable_rank_gamma,
    get_adapter_param_groups,
    insert_mode_adapters,
)

# ---------------------------------------------------------------------------
# Token constants (Chameleon / Anole)
# ---------------------------------------------------------------------------
PAD_TOKEN_ID = 1
SEP_TOKEN_ID = 8710          # <reserved08706> — separates prompt from answer
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2
IMAGE_TOKEN_MIN = 4
IMAGE_TOKEN_MAX = 8195


# ===========================================================================
# Datasets
# ===========================================================================

class VQATokenizedDataset(Dataset):
    """Loads pre-tokenised VQA jsonl (tokens field).

    Each sample is a 1-D LongTensor of token ids.
    Labels are built in the collate function (same as existing data.py).
    """

    def __init__(self, filepath: str, max_length: int = 2048):
        self.data: List[torch.LongTensor] = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                tokens = obj["tokens"]
                if isinstance(tokens, str):
                    tokens = json.loads(tokens)
                self.data.append(
                    torch.tensor(tokens, dtype=torch.long)[:max_length]
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx],)


def vqa_collate_fn(batch):
    """Collate for VQA: pad, build attention mask, mask labels before SEP."""
    seqs = [item[0] for item in batch]
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=PAD_TOKEN_ID)

    labels_list = []
    for seq in seqs:
        sep_pos = (seq == SEP_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            cut = sep_pos[0].item() + 1          # include SEP in masked region
            label = torch.cat([
                torch.full((cut,), -100, dtype=torch.long),
                seq[cut:],
            ])
        else:
            label = seq.clone()
        labels_list.append(label)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    attention_mask = (input_ids != PAD_TOKEN_ID).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class LaionKDDataset(Dataset):
    """LAION reference subset for KD.

    Each sample is constructed as [BOS, BOI, img_tok_1..1024, EOI] from the
    ``image_tokens`` field ([BOI, img_tok_1..1024, EOI] = 1026 tokens).
    This gives the model a pure image generation context; image-token
    next-token predictions are compared between student and frozen teacher.
    """

    def __init__(self, filepath: str, max_samples: Optional[int] = None):
        self.seqs: List[torch.LongTensor] = []
        with jsonlines.open(filepath) as reader:
            for idx, obj in enumerate(reader):
                if max_samples is not None and idx >= max_samples:
                    break
                image_tokens = obj.get("image_tokens", [])
                if isinstance(image_tokens, str):
                    image_tokens = json.loads(image_tokens)
                if not image_tokens:
                    continue
                # Prepend BOS; image_tokens already starts with BOI (8197)
                seq = [BOS_TOKEN_ID] + list(image_tokens)
                self.seqs.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        return (self.seqs[idx],)


def laion_collate_fn(batch):
    seqs = [item[0] for item in batch]
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask = (input_ids != PAD_TOKEN_ID).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def cycle(loader: DataLoader) -> Iterator:
    """Infinite iterator over a DataLoader."""
    while True:
        yield from loader


# ===========================================================================
# KD loss
# ===========================================================================

def compute_kd_loss(
    student_logits: torch.Tensor,   # [B, S, V]
    teacher_logits: torch.Tensor,   # [B, S, V]
    input_ids: torch.LongTensor,    # [B, S]
    beta: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL divergence on image-token prediction positions.

    Follows paper Eq (10)-(12):
        L_KD = beta^2 * KL( softmax(z_t / beta) || softmax(z_s / beta) )
    restricted to positions where the *next* token is an image token.
    Only the image-token logit slice [IMAGE_TOKEN_MIN : IMAGE_TOKEN_MAX+1]
    is used for efficiency.
    """
    # Shifted: position i predicts token i+1
    labels = input_ids[:, 1:]                   # [B, S-1]
    img_label_mask = (labels >= IMAGE_TOKEN_MIN) & (labels <= IMAGE_TOKEN_MAX)  # [B, S-1]

    if not img_label_mask.any():
        return student_logits.new_zeros(1).squeeze()

    s_log = student_logits[:, :-1, :]           # [B, S-1, V]
    t_log = teacher_logits[:, :-1, :]           # [B, S-1, V]

    # Restrict to image-token prediction positions
    s_img = s_log[img_label_mask]               # [N, V]
    t_img = t_log[img_label_mask]               # [N, V]

    # Restrict vocab slice to image token range
    s_img = s_img[:, IMAGE_TOKEN_MIN: IMAGE_TOKEN_MAX + 1].float()
    t_img = t_img[:, IMAGE_TOKEN_MIN: IMAGE_TOKEN_MAX + 1].float()

    t_probs   = F.softmax(t_img / beta, dim=-1)
    s_logprob = F.log_softmax(s_img / beta, dim=-1)

    kd = F.kl_div(s_logprob, t_probs, reduction="batchmean")
    return (beta ** 2) * kd.clamp(min=0.0)


# ===========================================================================
# Model helpers
# ===========================================================================

def load_model(ckpt: str, device: torch.device) -> ChameleonForConditionalGeneration:
    model = ChameleonForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16
    )
    model.to(device)
    return model


def save_adapters(model: nn.Module, save_dir: str) -> None:
    """Save only the MoDE adapter state dicts."""
    os.makedirs(save_dir, exist_ok=True)
    adapter_state: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapter_state[name] = param.detach().cpu()
    torch.save(adapter_state, os.path.join(save_dir, "mode_adapters.pt"))
    print(f"[MoDE] Adapters saved → {save_dir}/mode_adapters.pt  "
          f"({len(adapter_state)} tensors)")


def load_adapters(model: nn.Module, adapter_path: str, device: torch.device) -> None:
    state = torch.load(adapter_path, map_location=device)
    missing, unexpected = [], []
    model_state = dict(model.named_parameters())
    for k, v in state.items():
        if k in model_state:
            model_state[k].data.copy_(v.to(device))
        else:
            unexpected.append(k)
    for k in model_state:
        if model_state[k].requires_grad and k not in state:
            missing.append(k)
    if missing:
        print(f"[MoDE] WARNING: missing adapter keys: {missing[:5]}...")
    if unexpected:
        print(f"[MoDE] WARNING: unexpected adapter keys: {unexpected[:5]}...")
    print(f"[MoDE] Adapters loaded from {adapter_path}")


# ===========================================================================
# LCE bucket boundary computation
# ===========================================================================

def _instruction_length(tokens: List[int]) -> int:
    """Count instruction text tokens (before first SEP, no image/PAD/BOS)."""
    sep_pos = len(tokens)
    for i, t in enumerate(tokens):
        if t == SEP_TOKEN_ID:
            sep_pos = i
            break
    count = 0
    for t in tokens[:sep_pos]:
        if t == PAD_TOKEN_ID or t == BOS_TOKEN_ID:
            continue
        if IMAGE_TOKEN_MIN <= t <= IMAGE_TOKEN_MAX:
            continue
        count += 1
    return count


def compute_bucket_boundaries(
    data_dir: str,
    task_dirs: List[str],
    train_file: str = "train_data.jsonl",
    num_buckets: int = 4,
    max_length: int = 2048,
) -> List[int]:
    """Compute LCE bucket boundaries from training data instruction lengths.

    Scans all task training files, measures instruction text token length for
    every sample, then returns (num_buckets - 1) quantile values that split
    the distribution into ``num_buckets`` equal-frequency bins.

    For the default num_buckets=4 (short / medium / long / xlong) this
    returns [Q25, Q50, Q75] of the instruction-length distribution.

    Returns:
        Sorted list of (num_buckets - 1) integer boundary values.
    """
    lengths: List[int] = []
    for task_dir in task_dirs:
        fpath = os.path.join(data_dir, task_dir, train_file)
        if not os.path.exists(fpath):
            continue
        with jsonlines.open(fpath) as reader:
            for obj in reader:
                tokens = obj.get("tokens", [])
                if isinstance(tokens, str):
                    tokens = json.loads(tokens)
                tokens = tokens[:max_length]
                lengths.append(_instruction_length(tokens))

    if not lengths:
        raise ValueError(
            "No training samples found — cannot compute bucket boundaries. "
            f"Looked in: {[os.path.join(data_dir, d) for d in task_dirs]}"
        )

    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    qs = torch.linspace(0.0, 1.0, num_buckets + 1)[1:-1]  # (num_buckets-1) quantiles
    boundaries = [int(torch.quantile(lengths_t, q.item()).item()) for q in qs]

    # Deduplicate and sort (can happen if distribution is very concentrated)
    boundaries = sorted(set(boundaries))
    if len(boundaries) < num_buckets - 1:
        # Pad with values beyond max to fill remaining slots
        max_len = int(lengths_t.max().item()) + 1
        while len(boundaries) < num_buckets - 1:
            boundaries.append(max_len + len(boundaries))

    return boundaries


# ===========================================================================
# Training helpers
# ===========================================================================

def count_steps(dataset_len: int, batch_size: int, grad_acc: int, epochs: int) -> int:
    steps_per_epoch = (dataset_len + batch_size - 1) // batch_size
    return (steps_per_epoch // grad_acc) * epochs


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    total_steps: int,
    warmup_ratio: float,
    weight_decay: float = 0.01,
):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


# ===========================================================================
# Per-task training
# ===========================================================================

def train_one_task(
    task_name: str,
    student,
    teacher,
    mode_context: Dict,
    train_loader: DataLoader,
    laion_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    lambda_kd: float,
    beta_kd: float,
    grad_acc: int,
    log_steps: int,
    is_main: bool = True,
    use_gate: bool = False,
    bucket_boundaries: Optional[torch.Tensor] = None,
    teacher_device: Optional[torch.device] = None,
    use_bilce: bool = False,
    bilce_short_rank: Optional[int] = None,
    bilce_long_rank: Optional[int] = None,
) -> None:
    # When teacher_device is not set, it lives on the same device as the student
    if teacher_device is None:
        teacher_device = device
    student.train()
    teacher.eval()

    # Convenience: unwrapped model for attribute access
    raw_student = student.module if isinstance(student, DDP) else student

    total_loss_ce = 0.0
    total_loss_kd = 0.0
    step = 0

    for batch_idx, vqa_batch in enumerate(train_loader):
        input_ids      = vqa_batch["input_ids"].to(device)       # [B, S]
        attention_mask = vqa_batch["attention_mask"].to(device)
        labels         = vqa_batch["labels"].to(device)

        # ----------------------------------------------------------------
        # Build image mask for the VQA batch
        # ----------------------------------------------------------------
        vqa_img_mask = build_image_mask(input_ids)               # [B, S]

        # ================================================================
        # 1. CE Loss  — both T-MoE and V-Adapter receive gradients
        # ================================================================

        # LCE/BiLCE gate: compute sample-level γ before forward
        if use_bilce and bilce_short_rank is not None:
            gamma = compute_stable_rank_gamma(
                raw_student, input_ids,
                short_rank=bilce_short_rank,
                long_rank=bilce_long_rank,
                sep_token_id=SEP_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
                bos_token_id=BOS_TOKEN_ID,
                instruction_only=True,
            )
            mode_context["gamma"] = gamma
        elif use_gate and bucket_boundaries is not None:
            gamma = compute_lce_gamma(
                raw_student, input_ids, bucket_boundaries,
                sep_token_id=SEP_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
                bos_token_id=BOS_TOKEN_ID,
                instruction_only=True,
            )
            mode_context["gamma"] = gamma
        else:
            mode_context["gamma"] = None

        mode_context["image_mask"] = vqa_img_mask
        mode_context["kd_mode"]    = False
        raw_student.mode = "train-text"

        ce_out  = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        L_ce = ce_out.loss / grad_acc
        L_ce.backward()

        # ================================================================
        # 2. KD Loss  — only V-Adapter receives gradients (kd_mode=True
        #               means T-MoE branch is skipped in every MLP layer)
        # ================================================================
        laion_batch = next(laion_iter)
        # Send laion ids to teacher device for teacher forward
        laion_ids_teacher = laion_batch["input_ids"].to(teacher_device)
        # Send laion ids to student device for student forward + KD loss
        laion_ids_student = laion_batch["input_ids"].to(device)

        laion_img_mask = build_image_mask(laion_ids_student)

        with torch.no_grad():
            teacher.mode = "train-all"
            t_out    = teacher(input_ids=laion_ids_teacher)
            # Move teacher logits to student device before computing KD loss
            t_logits = t_out.logits.to(device)                  # [B_kd, S_kd, V]

        mode_context["image_mask"] = laion_img_mask
        mode_context["kd_mode"]    = True
        mode_context["gamma"]      = None   # γ unused when kd_mode skips T-MoE
        raw_student.mode = "train-all"

        s_out    = student(input_ids=laion_ids_student)
        s_logits = s_out.logits                                  # [B_kd, S_kd, V]

        L_kd = compute_kd_loss(s_logits, t_logits, laion_ids_student, beta=beta_kd)
        (lambda_kd * L_kd / grad_acc).backward()

        # ================================================================
        # Gradient accumulation step
        # ================================================================
        if (batch_idx + 1) % grad_acc == 0:
            raw = student.module if isinstance(student, DDP) else student
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            total_loss_ce += L_ce.item() * grad_acc
            total_loss_kd += L_kd.item()

            if is_main and step % log_steps == 0:
                print(
                    f"  [{task_name}] step {step:5d} | "
                    f"L_CE={total_loss_ce/step:.4f}  "
                    f"L_KD={total_loss_kd/step:.4f}"
                )

    # Handle leftover accumulation steps
    remainder = len(train_loader) % grad_acc
    if remainder != 0:
        # Use the underlying model params when wrapped with DDP
        raw_params = (
            student.module.parameters()
            if isinstance(student, DDP)
            else student.parameters()
        )
        torch.nn.utils.clip_grad_norm_(
            [p for p in raw_params if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if is_main:
        print(
            f"[{task_name}] Done — "
            f"avg L_CE={total_loss_ce/max(step,1):.4f}  "
            f"avg L_KD={total_loss_kd/max(step,1):.4f}"
        )


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MoDE Continual VQA Training")

    p.add_argument("--ckpt",       required=True,  help="Path to Anole HF checkpoint")
    p.add_argument("--data_dir",   required=True,  help="Root of MoDE-official/data")
    p.add_argument("--output_dir", required=True,  help="Root output directory")

    p.add_argument("--tasks",     nargs="+", default=["scienceqa", "textvqa", "vizwiz"])
    p.add_argument("--task_dirs", nargs="+", default=["ScienceQA",  "TextVQA",  "VizWiz"])
    p.add_argument("--use_caption", action="store_true",
                   help="Use train_data_with_caption.jsonl instead of train_data.jsonl")

    p.add_argument("--laion_file", default=None,
                   help="Path to laion_data.jsonl (defaults to data_dir/laion_data.jsonl)")
    p.add_argument("--laion_max_samples", type=int, default=None,
                   help="Cap LAION reference set size (None = all ~5859 samples)")

    # MoDE hyperparameters
    p.add_argument("--lora_rank",   type=int,   default=8)
    p.add_argument("--lora_alpha",  type=int,   default=16)
    p.add_argument("--num_experts", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    p.add_argument("--lambda_kd",   type=float, default=0.3)
    p.add_argument("--beta_kd",     type=float, default=2.0)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--bs",          type=int,   default=1,  help="VQA batch size per device")
    p.add_argument("--kd_bs",       type=int,   default=1,  help="LAION KD batch size")
    p.add_argument("--grad_acc",    type=int,   default=1)
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--log_steps",   type=int,   default=50)

    # LCE-LoRA
    p.add_argument("--use_lce", action="store_true",
                   help="Replace T-MoE experts with LCE dual-branch LoRA")
    p.add_argument("--base_rank", type=int, default=6,
                   help="Rank of the LCE base branch (always active)")
    p.add_argument("--ext_rank", type=int, default=2,
                   help="Rank of the LCE extension branch (gated by gamma)")

    # BiLCE-LoRA
    p.add_argument("--use_bilce", action="store_true",
                   help="Replace T-MoE experts with BiLCE three-branch LoRA "
                        "(common + short-private + long-private)")
    p.add_argument("--common_rank", type=int, default=2,
                   help="Rank of the BiLCE common branch (always active)")
    p.add_argument("--short_rank", type=int, default=3,
                   help="Rank of the BiLCE short-private branch (gated by 1-gamma)")
    p.add_argument("--long_rank", type=int, default=3,
                   help="Rank of the BiLCE long-private branch (gated by gamma)")

    # Shared gate args (used by both LCE and BiLCE)
    p.add_argument("--num_buckets", type=int, default=4,
                   help="Number of length buckets (short/medium/long/xlong)")
    p.add_argument("--auto_bucket", action="store_true",
                   help="Auto-compute bucket boundaries from training data quartiles "
                        "(recommended; overrides --bucket_boundaries)")
    p.add_argument("--bucket_boundaries", type=int, nargs="*", default=None,
                   help="Manual bucket boundaries (K-1 ints). "
                        "Used only when --auto_bucket is NOT set. "
                        "Default when omitted: uniform [20,50,100] for K=4.")

    # Split-GPU: load student and teacher on separate GPUs
    p.add_argument("--student_gpu", type=int, default=None,
                   help="GPU id for the student model (overrides LOCAL_RANK-based device)")
    p.add_argument("--teacher_gpu", type=int, default=None,
                   help="GPU id for the teacher model (may differ from student)")

    # Resume
    p.add_argument("--resume_adapter", default=None,
                   help="Path to mode_adapters.pt to resume from")

    return p.parse_args()


def setup_distributed():
    """Initialise torch.distributed if launched via torchrun, else single-GPU."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def is_main_process(local_rank: int) -> bool:
    return local_rank == 0


def main():
    args = parse_args()

    if args.use_lce and args.use_bilce:
        raise ValueError("--use_lce and --use_bilce are mutually exclusive; pick one.")

    # ── Distributed setup ──────────────────────────────────────────────
    local_rank, world_size = setup_distributed()
    is_main = is_main_process(local_rank)

    # ------------------------------------------------------------------
    # Device assignment
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        # Split-GPU mode: student and teacher on different cards
        if args.student_gpu is not None and args.teacher_gpu is not None:
            student_device  = torch.device(f"cuda:{args.student_gpu}")
            teacher_device  = torch.device(f"cuda:{args.teacher_gpu}")
            device = student_device   # default device = student (for data, scheduler, etc.)
            if world_size > 1:
                raise ValueError(
                    "--student_gpu / --teacher_gpu split is not compatible with "
                    "torchrun multi-GPU (use single-process launch instead)."
                )
        else:
            device = torch.device(f"cuda:{local_rank}")
            student_device = device
            teacher_device = device
    else:
        device = student_device = teacher_device = torch.device("cpu")

    assert len(args.tasks) == len(args.task_dirs), \
        "--tasks and --task_dirs must have the same length"

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        if student_device != teacher_device:
            print(f"[MoDE] Split-GPU mode: student → {student_device}, teacher → {teacher_device}")

    # ------------------------------------------------------------------
    # Load student backbone + insert MoDE adapters
    # ------------------------------------------------------------------
    if is_main:
        print(f"[MoDE] Loading student backbone on {student_device} ...")
    student = load_model(args.ckpt, student_device)
    use_gate = args.use_lce or args.use_bilce
    mode_context = insert_mode_adapters(
        student,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_experts=args.num_experts,
        use_lce=args.use_lce,
        base_rank=args.base_rank if args.use_lce else None,
        ext_rank=args.ext_rank if args.use_lce else None,
        use_bilce=args.use_bilce,
        common_rank=args.common_rank if args.use_bilce else None,
        short_rank=args.short_rank if args.use_bilce else None,
        long_rank=args.long_rank if args.use_bilce else None,
        num_buckets=args.num_buckets,
    )

    if args.resume_adapter:
        load_adapters(student, args.resume_adapter, student_device)

    # Wrap with DDP when using multiple GPUs (only when NOT in split-GPU mode)
    if world_size > 1:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

    # ------------------------------------------------------------------
    # Load frozen teacher (no adapters)
    # ------------------------------------------------------------------
    if is_main:
        print(f"[MoDE] Loading frozen teacher on {teacher_device} ...")
    teacher = load_model(args.ckpt, teacher_device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ------------------------------------------------------------------
    # LAION reference DataLoader (cycles infinitely)
    # ------------------------------------------------------------------
    laion_path = args.laion_file or os.path.join(args.data_dir, "laion_data.jsonl")
    if is_main:
        print(f"[MoDE] Loading LAION reference set from {laion_path}...")
    laion_dataset = LaionKDDataset(laion_path, max_samples=args.laion_max_samples)
    if is_main:
        print(f"[MoDE] LAION reference set size: {len(laion_dataset)}")

    laion_sampler = DistributedSampler(laion_dataset, shuffle=True) if world_size > 1 else None
    laion_loader  = DataLoader(
        laion_dataset,
        batch_size=args.kd_bs,
        sampler=laion_sampler,
        shuffle=(laion_sampler is None),
        collate_fn=laion_collate_fn,
        drop_last=False,
        num_workers=0,
    )
    laion_iter = cycle(laion_loader)

    # ------------------------------------------------------------------
    # Gate bucket boundaries (LCE only; BiLCE uses parameter-free
    # Stable-Rank Prompt Gate and does not need bucket boundaries)
    # ------------------------------------------------------------------
    gate_tag = "BiLCE" if args.use_bilce else "LCE"
    if use_gate and not args.use_bilce:
        if args.auto_bucket:
            train_file_name = (
                "train_data_with_caption.jsonl" if args.use_caption else "train_data.jsonl"
            )
            if is_main:
                print(f"[{gate_tag}] Computing bucket boundaries from training data quartiles…")
            raw_boundaries = compute_bucket_boundaries(
                data_dir=args.data_dir,
                task_dirs=args.task_dirs,
                train_file=train_file_name,
                num_buckets=args.num_buckets,
                max_length=args.max_length,
            )
            bucket_json_path = os.path.join(args.output_dir, "lce_bucket_boundaries.json")
            if is_main:
                with open(bucket_json_path, "w") as _f:
                    json.dump(
                        {
                            "num_buckets": args.num_buckets,
                            "boundaries": raw_boundaries,
                            "labels": ["short", "medium", "long", "xlong"][: args.num_buckets],
                        },
                        _f,
                        indent=2,
                    )
                print(f"[{gate_tag}] boundaries saved → {bucket_json_path}")
        else:
            raw_boundaries = sorted(args.bucket_boundaries or [20, 50, 100])

        bucket_boundaries = torch.tensor(raw_boundaries, dtype=torch.long, device=device)
        labels = ["short", "medium", "long", "xlong"][: args.num_buckets]
        if is_main:
            for lo, hi, lbl in zip(
                [0] + raw_boundaries,
                raw_boundaries + ["∞"],
                labels,
            ):
                print(f"[{gate_tag}]   {lbl:8s}: [{lo}, {hi})")
    elif args.use_bilce:
        bucket_boundaries = None
        r_p = args.short_rank + args.long_rank
        if is_main:
            print(f"[BiLCE] Stable-Rank Prompt Gate: r_p = short_rank + long_rank = {r_p}")
            print(f"[BiLCE]   γ(x) = sr(H̃) / (sr(H̃) + {r_p})")
            print(f"[BiLCE]   H from frozen layer-0 pre-MLP hidden states")
    else:
        bucket_boundaries = None

    # ------------------------------------------------------------------
    # Sequential task training
    # ------------------------------------------------------------------
    train_file = "train_data_with_caption.jsonl" if args.use_caption else "train_data.jsonl"

    # Unwrap DDP to access mode_context / save adapters
    student_unwrapped = student.module if isinstance(student, DDP) else student

    for task_idx, (task_name, task_dir) in enumerate(zip(args.tasks, args.task_dirs)):
        train_path = os.path.join(args.data_dir, task_dir, train_file)
        if is_main:
            print(f"\n{'='*60}")
            print(f"[MoDE] Task {task_idx+1}/{len(args.tasks)}: {task_name}")
            print(f"  train data : {train_path}")
            print(f"{'='*60}")

        # Load VQA training data
        train_dataset = VQATokenizedDataset(train_path, max_length=args.max_length)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
        train_loader  = DataLoader(
            train_dataset,
            batch_size=args.bs,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=vqa_collate_fn,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
        )
        if is_main:
            print(f"[MoDE]   samples: {len(train_dataset)}  "
                  f"batches/rank: {len(train_loader)}  "
                  f"world_size: {world_size}")

        # Build optimizer + scheduler for this task
        # total_steps is computed per-rank (each rank sees 1/world_size of data)
        total_steps = count_steps(
            len(train_dataset) // world_size, args.bs, args.grad_acc, args.epochs
        )
        optimizer, scheduler = build_optimizer_and_scheduler(
            student_unwrapped, args.lr, total_steps, args.warmup_ratio
        )

        # Train for args.epochs (default 1) on this task
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if is_main:
                print(f"[MoDE] Epoch {epoch+1}/{args.epochs}")
            train_one_task(
                task_name    = task_name,
                student      = student,
                teacher      = teacher,
                mode_context = mode_context,
                train_loader = train_loader,
                laion_iter   = laion_iter,
                optimizer    = optimizer,
                scheduler    = scheduler,
                device       = device,
                lambda_kd    = args.lambda_kd,
                beta_kd      = args.beta_kd,
                grad_acc     = args.grad_acc,
                log_steps    = args.log_steps,
                is_main      = is_main,
                use_gate     = use_gate,
                bucket_boundaries = bucket_boundaries,
                teacher_device    = teacher_device,
                use_bilce         = args.use_bilce,
                bilce_short_rank  = args.short_rank if args.use_bilce else None,
                bilce_long_rank   = args.long_rank if args.use_bilce else None,
            )

        # Only rank-0 saves adapter weights
        if is_main:
            task_save_dir = os.path.join(args.output_dir, f"mode_{task_name}")
            save_adapters(student_unwrapped, task_save_dir)

        # Sync all ranks before next task
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print("\n[MoDE] All tasks complete!")
        print(f"  Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
