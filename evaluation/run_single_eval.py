#!/usr/bin/env python3
"""
Single-task evaluation: load one model checkpoint and evaluate on one dataset.

Examples:
    # Stage 2 (ImageNet weights) → evaluate ScienceQA
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_single_eval \
        --model_dir  /data1/data/kangborui/zhongyukun/medmax/output_dir/seqlora_imagenet/seqlora_imagenet_stage2 \
        --eval_task  scienceqa \
        --model_format hf \
        --image_base_dir /data1/data/kangborui/zhongyukun/medmax/coin_raw \
        --data_base_dir  /data1/data/kangborui/zhongyukun/medmax/MoDE-official \
        --hf_base_model_dir /data1/data/kangborui/zhongyukun/medmax/anole_7b_hf

    # Stage 2 (ImageNet weights) → evaluate ImageNet
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_single_eval \
        --model_dir  /data1/data/kangborui/zhongyukun/medmax/output_dir/seqlora_imagenet/seqlora_imagenet_stage2 \
        --eval_task  imagenet \
        --model_format hf \
        --image_base_dir /data1/data/kangborui/zhongyukun/medmax/coin_raw \
        --data_base_dir  /data1/data/kangborui/zhongyukun/medmax/MoDE-official \
        --hf_base_model_dir /data1/data/kangborui/zhongyukun/medmax/anole_7b_hf

    # Stage 3 (GQA weights, chameleon format) → evaluate ImageNet
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_single_eval \
        --model_dir  /data1/data/kangborui/zhongyukun/medmax/output_dir/seqlora_gqa/seqlora_gqa_stage3 \
        --eval_task  imagenet \
        --model_format chameleon \
        --image_base_dir /data1/data/kangborui/zhongyukun/medmax/coin_raw \
        --data_base_dir  /data1/data/kangborui/zhongyukun/medmax/MoDE-official

    # Evaluate multiple tasks at once
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.run_single_eval \
        --model_dir  /data1/data/kangborui/zhongyukun/medmax/output_dir/seqlora_imagenet/seqlora_imagenet_stage2 \
        --eval_task  scienceqa imagenet \
        --model_format hf \
        --image_base_dir /data1/data/kangborui/zhongyukun/medmax/coin_raw \
        --data_base_dir  /data1/data/kangborui/zhongyukun/medmax/MoDE-official \
        --hf_base_model_dir /data1/data/kangborui/zhongyukun/medmax/anole_7b_hf
"""

import argparse
import json
import os
import sys

import torch
from evaluation.run_continual_vqa_eval import (
    MAX_GEN_LEN,
    TEST_DATA_PATHS,
    compute_task_accuracy,
    load_model_chameleon,
    load_model_hf,
    load_test_data,
    run_inference_chameleon,
    run_inference_hf,
)
from evaluation.continual_metrics import save_json


VALID_TASKS = list(TEST_DATA_PATHS.keys())


def main():
    parser = argparse.ArgumentParser(description="Evaluate one model on one or more datasets")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model checkpoint (e.g. output_dir/seqlora_imagenet/seqlora_imagenet_stage2)")
    parser.add_argument("--eval_task", type=str, nargs="+", required=True, choices=VALID_TASKS,
                        help=f"Dataset(s) to evaluate on: {VALID_TASKS}")
    parser.add_argument("--model_format", type=str, required=True, choices=["hf", "chameleon"],
                        help="Model format: hf (stage2) or chameleon (stage3)")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Base directory for images (parent of ScienceQA/, ImageNet/, GQA/)")
    parser.add_argument("--data_base_dir", type=str, required=True,
                        help="MoDE-official directory with instructions/")
    parser.add_argument("--hf_base_model_dir", type=str, default=None,
                        help="HF base model dir for processor (required for hf format)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predictions (default: model_dir/single_eval)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit number of test samples (0 = all)")
    args = parser.parse_args()

    if args.model_format == "hf" and not args.hf_base_model_dir:
        parser.error("--hf_base_model_dir is required when --model_format=hf")

    output_dir = args.output_dir or os.path.join(args.model_dir, "single_eval")
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    print("=" * 60)
    print(f"Loading model: {args.model_dir}")
    print(f"Format: {args.model_format}")
    print(f"Tasks to evaluate: {args.eval_task}")
    print("=" * 60)

    if args.model_format == "chameleon":
        model = load_model_chameleon(args.model_dir)
        processor = None
    else:
        model, processor = load_model_hf(args.model_dir, args.hf_base_model_dir)

    # Evaluate each task
    all_results = {}
    for task in args.eval_task:
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {task}")
        print(f"{'─' * 50}")

        test_path = os.path.join(args.data_base_dir, TEST_DATA_PATHS[task])
        samples = load_test_data(test_path, args.image_base_dir, task, require_answer=True)

        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        max_gen = MAX_GEN_LEN.get(task, 20)

        if args.model_format == "chameleon":
            predictions = run_inference_chameleon(model, samples, task, max_gen_len=max_gen)
        else:
            predictions = run_inference_hf(model, processor, samples, task, max_gen_len=max_gen)

        acc, detailed = compute_task_accuracy(predictions)
        correct = sum(d["correct"] for d in detailed)
        total = len(detailed)

        print(f"\n  >>> {task} accuracy: {acc:.4f}  ({correct}/{total})")

        # Show first few predictions
        print(f"\n  Sample predictions (first 5):")
        for d in detailed[:5]:
            mark = "✓" if d["correct"] else "✗"
            print(f"    {mark}  answer={d['answer']!r:20s}  prediction={d['prediction']!r}")

        # Save predictions
        pred_path = os.path.join(output_dir, f"{task}_predictions.json")
        save_json(pred_path, detailed)
        print(f"  Predictions saved to {pred_path}")

        all_results[task] = acc

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_dir}")
    for task, acc in all_results.items():
        print(f"  {task:15s}: {acc:.4f}")
    if len(all_results) > 1:
        avg = sum(all_results.values()) / len(all_results)
        print(f"  {'average':15s}: {avg:.4f}")

    summary_path = os.path.join(output_dir, "eval_summary.json")
    save_json(summary_path, {"model_dir": args.model_dir, "results": all_results})
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
