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
) -> None:
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
        laion_ids   = laion_batch["input_ids"].to(device)        # [B_kd, S_kd]

        laion_img_mask = build_image_mask(laion_ids)

        with torch.no_grad():
            teacher.mode = "train-all"
            t_out       = teacher(input_ids=laion_ids)
            t_logits    = t_out.logits                           # [B_kd, S_kd, V]

        mode_context["image_mask"] = laion_img_mask
        mode_context["kd_mode"]    = True
        raw_student.mode = "train-all"

        s_out    = student(input_ids=laion_ids)
        s_logits = s_out.logits                                  # [B_kd, S_kd, V]

        L_kd = compute_kd_loss(s_logits, t_logits, laion_ids, beta=beta_kd)
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

    # ── Distributed setup ──────────────────────────────────────────────
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process(local_rank)

    assert len(args.tasks) == len(args.task_dirs), \
        "--tasks and --task_dirs must have the same length"

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load student backbone + insert MoDE adapters
    # ------------------------------------------------------------------
    if is_main:
        print("[MoDE] Loading student backbone...")
    student = load_model(args.ckpt, device)
    mode_context = insert_mode_adapters(
        student,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_experts=args.num_experts,
    )

    if args.resume_adapter:
        load_adapters(student, args.resume_adapter, device)

    # Wrap with DDP when using multiple GPUs
    if world_size > 1:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

    # ------------------------------------------------------------------
    # Load frozen teacher (no adapters) — each rank holds its own copy
    # ------------------------------------------------------------------
    if is_main:
        print("[MoDE] Loading frozen teacher (same checkpoint, no adapters)...")
    teacher = load_model(args.ckpt, device)
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
