#!/usr/bin/env python3
"""
Continual Learning Evaluation for VQA Tasks (ScienceQA -> TextVQA -> GQA).

For each stage i (after training task i), loads the model and evaluates on
tasks 1..i.  Builds an accuracy matrix, then computes:
  - per-stage ACC / Fgt
  - final ACC  = mean accuracy on all tasks after the last stage
  - final Fgt  = mean forgetting across non-final tasks

Supports two model formats:
  * chameleon  – the native Chameleon checkpoint (/output_dir/seqlora_?/seqlora_?_stage3)
  * hf         – HuggingFace ChameleonForConditionalGeneration (/output_dir/seqlora_?/seqlora_?_stage2)

Test data is loaded from MoDE-official/instructions/{TaskName}/test.json.
For TextVQA, whose test.json has no ground-truth answers, the script falls
back to MoDE-official/data/TextVQA/val_data.jsonl (which has answers).

Usage (from the medmax/ directory):
    # Chameleon format (stage3)
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_continual_vqa_eval \\
        --model_base_dir  /data1/data/kangborui/zhongyukun/medmax/output_dir \\
        --image_base_dir  /path/to/cl_datasets \\
        --data_base_dir   /data1/data/kangborui/zhongyukun/medmax/MoDE-official \\
        --output_dir      /data1/data/kangborui/zhongyukun/medmax/output_dir/continual_eval_results \\
        --model_format    chameleon

    # HF format (stage2)
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_continual_vqa_eval \\
        --model_base_dir  /data1/data/kangborui/zhongyukun/medmax/output_dir \\
        --image_base_dir  /path/to/cl_datasets \\
        --data_base_dir   /data1/data/kangborui/zhongyukun/medmax/MoDE-official \\
        --output_dir      /data1/data/kangborui/zhongyukun/medmax/output_dir/continual_eval_results \\
        --model_format    hf \\
        --hf_base_model_dir /data1/data/kangborui/zhongyukun/medmax/anole_7b_hf
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from evaluation.continual_metrics import (
    calculate_accuracy,
    compute_acc_fgt,
    compute_bwt,
    compute_stage_acc_fgt,
    exact_match,
    format_matrix,
    init_accuracy_matrix,
    load_json,
    normalize_answer,
    save_json,
    update_accuracy_matrix,
)


# ──────────────────────────────────────────────────────────────────────
# Default task configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_TASK_ORDER = ["scienceqa", "imagenet", "gqa"]

TASK_DIR_NAMES = {
    "scienceqa": "scienceqa",
    "imagenet":  "imagenet",
    "gqa":       "gqa",
    "textvqa":   "textvqa",
    "vizwiz":    "vizwiz",
}

# Instruction names in MoDE-official/instructions/
TASK_INSTRUCTION_NAMES = {
    "scienceqa": "ScienceQA",
    "imagenet":  "ImageNet",
    "gqa":       "GQA",
    "textvqa":   "TextVQA",
    "vizwiz":    "VizWiz",
}

# Test data files (relative to data_base_dir)
# For ScienceQA/ImageNet/GQA: use instructions/ test.json (has labels)
# For TextVQA/VizWiz: use data/ test_data.jsonl (repurposed val set with labels)
TEST_DATA_PATHS = {
    "scienceqa": "instructions/ScienceQA/test.json",
    "imagenet":  "instructions/ImageNet/test.json",
    "gqa":       "instructions/GQA/test.json",
    "textvqa":   "data/TextVQA/test_data.jsonl",
    "vizwiz":    "data/VizWiz/test_data.jsonl",
}

# Maximum generation length for each task
MAX_GEN_LEN = {
    "scienceqa": 5,    # letter answer (A/B/C/D)
    "imagenet":  30,   # short phrase
    "gqa":       20,   # short phrase
    "textvqa":   20,   # short phrase
    "vizwiz":    20,   # short phrase
}


def get_model_path(
    model_base_dir: str,
    task: str,
    model_format: str,
    ckpt_prefix: str = "seqlora",
) -> str:
    """Build model checkpoint path for a given task and format.

    ckpt_prefix="seqlora" (default) → seqlora_{task}/seqlora_{task}_stage{2|3}
    ckpt_prefix="mode"              → mode_{task}/hf_model
    """
    dirname = TASK_DIR_NAMES[task]
    if ckpt_prefix == "mode":
        return os.path.join(model_base_dir, f"mode_{dirname}")
    stage = "stage3" if model_format == "chameleon" else "stage2"
    return os.path.join(
        model_base_dir,
        f"{ckpt_prefix}_{dirname}",
        f"{ckpt_prefix}_{dirname}_{stage}",
    )


def get_test_data_path(data_base_dir: str, task: str) -> str:
    """Resolve the test data path for a given task."""
    return os.path.join(data_base_dir, TEST_DATA_PATHS[task])


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _strip_sft_answer(text: str) -> str:
    """Remove answer appended after <reserved08706> in tokenized data."""
    sep = "<reserved08706>"
    if sep in text:
        return text.split(sep)[0].strip()
    return text.strip()


def load_test_data(
    data_path: str,
    image_base_dir: str,
    task_name: str,
) -> List[Dict[str, Any]]:
    """
    Load test samples from either a JSON array file or JSONL file.
    Returns list of dicts with keys: question_id, text, answer, image (abs path or None).
    """
    samples: List[Dict[str, Any]] = []

    if data_path.endswith(".jsonl"):
        # JSONL format (e.g. data/TextVQA/val_data.jsonl)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                question = _strip_sft_answer(obj.get("text", ""))
                answer = obj.get("answer", "")
                # Image path: remap from original absolute to local base
                raw_img = obj.get("image_path", "")
                if raw_img:
                    raw_img = raw_img.replace("\\", "")
                    # Extract relative portion: e.g. TextVQA/train_images/xxx.jpg
                    # Original paths look like /home/xw6956/CoIN/cl_datasets/TextVQA/...
                    match = re.search(
                        r"(ScienceQA|TextVQA|GQA|VizWiz|ImageNet)/(.*)", raw_img
                    )
                    if match:
                        rel = match.group(0)
                        img_path = os.path.join(image_base_dir, rel)
                    else:
                        img_path = raw_img  # fallback
                else:
                    img_path = None

                samples.append({
                    "question_id": obj.get("question_id", len(samples)),
                    "text": question,
                    "answer": answer,
                    "image": img_path,
                })
    else:
        # JSON array format (instructions/*.json)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data:
            answer = obj.get("answer", "")
            if not answer:
                continue  # skip samples without ground-truth
            raw_img = obj.get("image", "")
            if raw_img:
                # Normalise relative path
                raw_img = raw_img.lstrip("./")
                img_path = os.path.join(image_base_dir, raw_img)
            else:
                img_path = None
            samples.append({
                "question_id": obj.get("question_id", len(samples)),
                "text": obj.get("text", ""),
                "answer": answer,
                "image": img_path,
            })

    print(f"  [{task_name}] Loaded {len(samples)} test samples from {data_path}")
    return samples


# ──────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────

def load_model_chameleon(model_dir: str):
    """Load a Chameleon-format model for inference."""
    from inference.inference_utils import load_chameleon
    print(f"  Loading Chameleon model from {model_dir} ...")
    model = load_chameleon(model_dir)
    return model


def load_model_hf(model_dir: str, processor_dir: str):
    """Load an HF-format model + processor for inference."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import ChameleonForConditionalGeneration
    from transformers import ChameleonProcessor

    print(f"  Loading HF model from {model_dir} ...")
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = ChameleonProcessor.from_pretrained(processor_dir)
    return model, processor


def load_model_mode(
    adapter_dir: str,
    base_model_dir: str,
    processor_dir: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    num_experts: int = 4,
):
    """Load base Anole model, insert MoDE adapters, load adapter weights.

    MoDE saves adapters as mode_adapters.pt (only trainable params).
    The base backbone weights come from the original Anole checkpoint.
    A forward pre-hook auto-sets image_mask from input_ids on every call
    (including each step of generate()).
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
    from training.mode_adapters import insert_mode_adapters, build_image_mask
    from training.mode_train import load_adapters

    adapter_path = os.path.join(adapter_dir, "mode_adapters.pt")
    print(f"  Loading base model from {base_model_dir} ...")
    print(f"  Loading MoDE adapters from {adapter_path} ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChameleonForConditionalGeneration.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)

    mode_context = insert_mode_adapters(
        model, lora_rank=lora_rank, lora_alpha=lora_alpha, num_experts=num_experts,
    )
    load_adapters(model, adapter_path, device)
    model.eval()
    model.mode = "inference-text"
    mode_context["kd_mode"] = False

    def _auto_set_image_mask(module, args, kwargs):
        """Pre-hook: derive image_mask from input_ids before each forward."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is not None:
            mode_context["image_mask"] = build_image_mask(input_ids)
        else:
            mode_context["image_mask"] = None

    model.register_forward_pre_hook(_auto_set_image_mask, with_kwargs=True)

    processor = ChameleonProcessor.from_pretrained(processor_dir)
    return model, processor


# ──────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────

def run_inference_chameleon(
    model,
    samples: List[Dict[str, Any]],
    task_name: str,
    max_gen_len: int = 20,
) -> List[Dict[str, Any]]:
    """Run inference with a Chameleon-format model and return predictions."""
    from inference.inference_utils import chameleon_generate

    results = []
    for sample in tqdm(samples, desc=f"  Inference [{task_name}]"):
        question = sample["text"]
        image_path = sample["image"]

        # Build content list
        if image_path and os.path.isfile(image_path):
            content = [image_path, question]
            modality = ["image", "text"]
        else:
            content = [question]
            modality = ["text"]

        try:
            generated = chameleon_generate(
                model,
                content=content,
                modality=modality,
                task="text-gen",
                sft=False,
                max_gen_len=max_gen_len,
                greedy=True,
            )
            pred_text = generated[0] if generated else ""
        except Exception as e:
            print(f"    [WARN] Inference failed for qid={sample['question_id']}: {e}")
            pred_text = ""

        results.append({
            "question_id": sample["question_id"],
            "answer": sample["answer"],
            "prediction": pred_text,
        })
    return results


def run_inference_hf(
    model,
    processor,
    samples: List[Dict[str, Any]],
    task_name: str,
    max_gen_len: int = 20,
) -> List[Dict[str, Any]]:
    """Run inference with an HF-format model and return predictions."""
    results = []
    for sample in tqdm(samples, desc=f"  Inference [{task_name}]"):
        question = sample["text"]
        image_path = sample["image"]

        try:
            if image_path and os.path.isfile(image_path):
                image = Image.open(image_path).convert("RGB")
                # Insert <image> placeholder if not present
                if "<image>" not in question:
                    prompt = f"<image>{question}"
                else:
                    prompt = question
                inputs = processor(
                    images=[image],
                    text=prompt,
                    return_tensors="pt",
                ).to(device=model.device, dtype=model.dtype)
            else:
                inputs = processor(text=question, return_tensors="pt").to(device=model.device, dtype=model.dtype)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_gen_len,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            pred_text = processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"    [WARN] HF inference failed for qid={sample['question_id']}: {e}")
            pred_text = ""

        results.append({
            "question_id": sample["question_id"],
            "answer": sample["answer"],
            "prediction": pred_text,
        })
    return results


# ──────────────────────────────────────────────────────────────────────
# Accuracy computation
# ──────────────────────────────────────────────────────────────────────

def compute_task_accuracy(
    predictions: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute exact-match accuracy from a list of prediction dicts."""
    scores = []
    detailed = []
    for pred in predictions:
        score = exact_match(str(pred["prediction"]), str(pred["answer"]))
        scores.append(score)
        detailed.append({**pred, "correct": score})
    acc = calculate_accuracy(scores)
    return acc, detailed


# ──────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────

def run_continual_eval(args) -> Dict[str, Any]:
    """
    Main continual learning evaluation.

    For each stage i = 1..N (task order: scienceqa, textvqa, gqa):
      1. Load model trained after task i
      2. Evaluate on tasks 1..i
      3. Record accuracy in matrix[i][j]
    Finally compute ACC and Fgt.
    """
    tasks = args.task_order if hasattr(args, 'task_order') and args.task_order else DEFAULT_TASK_ORDER
    num_tasks = len(tasks)

    # Resolve paths
    data_base = args.data_base_dir
    model_base = args.model_base_dir
    image_base = args.image_base_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    matrix_path = os.path.join(output_dir, "accuracy_matrix.json")

    # Pre-load all test data
    print("=" * 60)
    print("Loading test data for all tasks ...")
    print("=" * 60)
    test_data: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        test_path = get_test_data_path(data_base, task)
        test_data[task] = load_test_data(
            test_path, image_base, task
        )

    # Iterate over stages
    all_stage_results: Dict[str, Dict[str, float]] = {}

    for stage_idx in range(num_tasks):
        stage = stage_idx + 1  # 1-based
        current_task = tasks[stage_idx]
        model_dir = get_model_path(model_base, current_task, args.model_format,
                                   ckpt_prefix=args.ckpt_prefix)

        print()
        print("=" * 60)
        print(f"Stage {stage}/{num_tasks}: model trained after '{current_task}'")
        print(f"  Model path: {model_dir}")
        print(f"  Evaluating on tasks: {tasks[:stage]}")
        print("=" * 60)

        # Load model
        model_obj = None
        processor_obj = None
        if args.model_format == "chameleon":
            model_obj = load_model_chameleon(model_dir)
        elif args.ckpt_prefix == "mode":
            processor_dir = args.processor_dir or args.hf_base_model_dir
            model_obj, processor_obj = load_model_mode(
                adapter_dir=model_dir,
                base_model_dir=args.hf_base_model_dir,
                processor_dir=processor_dir,
                lora_rank=args.mode_lora_rank,
                lora_alpha=args.mode_lora_alpha,
                num_experts=args.mode_num_experts,
            )
        else:
            processor_dir = args.processor_dir or args.hf_base_model_dir
            model_obj, processor_obj = load_model_hf(model_dir, processor_dir)

        # Evaluate on tasks 1..stage
        stage_results: Dict[str, float] = {}

        for eval_task_idx in range(stage):
            eval_task = tasks[eval_task_idx]

            print(f"\n  Evaluating task '{eval_task}' ...")
            samples = test_data[eval_task]
            max_gen = MAX_GEN_LEN.get(eval_task, 20)

            if args.max_samples > 0:
                samples = samples[: args.max_samples]

            if args.model_format == "chameleon":
                predictions = run_inference_chameleon(
                    model_obj, samples, eval_task, max_gen_len=max_gen,
                )
            else:
                predictions = run_inference_hf(
                    model_obj, processor_obj, samples, eval_task, max_gen_len=max_gen,
                )

            pred_dir = os.path.join(output_dir, f"stage{stage}_predictions")
            os.makedirs(pred_dir, exist_ok=True)

            acc, detailed = compute_task_accuracy(predictions)
            stage_results[eval_task] = acc
            print(f"    -> {eval_task} accuracy: {acc:.4f}  ({sum(d['correct'] for d in detailed)}/{len(detailed)})")
            pred_path = os.path.join(pred_dir, f"{eval_task}_predictions.json")
            save_json(pred_path, detailed)

        # Update accuracy matrix
        matrix_data = update_accuracy_matrix(matrix_path, stage, tasks, stage_results)
        all_stage_results[f"stage{stage}"] = stage_results

        # Compute stage-level metrics (skips None entries from submission-only tasks)
        matrix = matrix_data["matrix"]
        acc_stage, fgt_stage = compute_stage_acc_fgt(matrix, stage)
        print(f"\n  Stage {stage} metrics: ACC_stage={acc_stage:.4f}, Fgt_stage={fgt_stage:.4f}")

        # Release model to free GPU memory before loading next
        del model_obj
        del processor_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final metrics ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CONTINUAL LEARNING RESULTS")
    print("=" * 60)

    matrix_data = load_json(matrix_path)
    matrix = matrix_data["matrix"]

    # Print accuracy matrix
    print(format_matrix(tasks, matrix))

    # Final ACC, Fgt and BWT
    acc, fgt = compute_acc_fgt(matrix)
    bwt = compute_bwt(matrix)
    print(f"\nFinal ACC (average accuracy after last stage): {acc:.4f}")
    print(f"Final Fgt (average forgetting):                 {fgt:.4f}")
    print(f"Final BWT (backward transfer):                  {bwt:.4f}")

    # Per-stage metrics
    print("\nPer-stage breakdown:")
    for s in range(1, num_tasks + 1):
        acc_s, fgt_s = compute_stage_acc_fgt(matrix, s)
        print(f"  Stage {s}: ACC={acc_s:.4f}, Fgt={fgt_s:.4f}")

    # Save summary
    summary = {
        "task_order": tasks,
        "accuracy_matrix": matrix,
        "final_ACC": acc,
        "final_Fgt": fgt,
        "final_BWT": bwt,
        "per_stage": {},
    }
    for s in range(1, num_tasks + 1):
        acc_s, fgt_s = compute_stage_acc_fgt(matrix, s)
        summary["per_stage"][f"stage{s}"] = {
            "ACC": acc_s,
            "Fgt": fgt_s,
            "task_accuracies": {
                tasks[j]: matrix[s - 1][j]
                for j in range(s)
                if matrix[s - 1][j] is not None
            },
        }
    summary_path = os.path.join(output_dir, "continual_metrics_summary.json")
    save_json(summary_path, summary)
    print(f"\nSummary saved to {summary_path}")

    return summary


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continual learning VQA evaluation (ScienceQA -> TextVQA -> GQA)"
    )
    parser.add_argument(
        "--model_base_dir",
        type=str,
        default="../output_dir",
        help="Base directory containing model checkpoints (seqlora_*/)",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        required=True,
        help="Base directory for images (parent of ScienceQA/, TextVQA/, GQA/ image dirs)",
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="../MoDE-official",
        help="MoDE-official directory with instructions/ and data/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output_dir/continual_eval_results",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        choices=["chameleon", "hf"],
        default="chameleon",
        help="Model format to use: 'chameleon' (stage3) or 'hf' (stage2)",
    )
    parser.add_argument(
        "--hf_base_model_dir",
        type=str,
        default="../anole_7b_hf",
        help="Path to HF base model / processor (used for HF format tokenizer/processor)",
    )
    parser.add_argument(
        "--processor_dir",
        type=str,
        default="",
        help="Path to ChameleonProcessor dir (defaults to hf_base_model_dir if empty)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max samples per task for debugging (0 = all samples)",
    )
    parser.add_argument(
        "--task_order",
        nargs="+",
        default=None,
        help="Ordered list of tasks, e.g. scienceqa textvqa vizwiz (default: scienceqa imagenet gqa)",
    )
    parser.add_argument(
        "--ckpt_prefix",
        type=str,
        default="seqlora",
        help=(
            "Checkpoint directory prefix. "
            "'seqlora' (default): {base}/seqlora_{task}/seqlora_{task}_stage{2|3}. "
            "'mode': {base}/mode_{task}/ (loads base model + adapter weights)."
        ),
    )
    parser.add_argument("--mode_lora_rank",   type=int, default=8,  help="MoDE LoRA rank (must match training)")
    parser.add_argument("--mode_lora_alpha",  type=int, default=16, help="MoDE LoRA alpha (must match training)")
    parser.add_argument("--mode_num_experts", type=int, default=4,  help="MoDE num experts (must match training)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_continual_eval(args)
