"""
Continual learning evaluation tracker.

Supports three modes for feeding per-task accuracy:
  1. --results_json        : pre-computed {task: accuracy} mapping
  2. --predictions_dir     : directory of per-task prediction files (JSONL or JSON)
  3. --run_inference        : run model inference directly via run_continual_vqa_eval

The script updates an accuracy matrix, computes per-stage and final ACC / Fgt,
and optionally saves all metrics to a JSON file.

Typical CLI usage (post-hoc, after inference has already been done):
    python -m evaluation.continual_eval \
        --tasks scienceqa,textvqa,gqa \
        --stage 3 \
        --matrix_path results/accuracy_matrix.json \
        --predictions_dir results/stage3_predictions \
        --predictions_suffix _predictions.json \
        --answer_key answer \
        --prediction_key prediction \
        --metrics_path results/metrics.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional

from evaluation.continual_metrics import (
    calculate_accuracy,
    compute_acc_fgt,
    compute_stage_acc_fgt,
    exact_match,
    load_json,
    save_json,
    update_accuracy_matrix,
)


def parse_tasks(tasks: str) -> List[str]:
    return [task.strip() for task in tasks.split(",") if task.strip()]


def load_results_json(path: str) -> Dict[str, float]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Results JSON must be a mapping of task name to accuracy.")
    return {str(key): float(value) for key, value in data.items()}


def compute_exact_match_from_predictions(
    predictions_dir: str,
    tasks: List[str],
    answer_key: str,
    prediction_key: str,
    suffix: str,
) -> Dict[str, float]:
    """Compute exact-match accuracy from per-task prediction files.

    Supports both JSONL (one JSON object per line) and plain JSON
    (a list of objects) files.
    """
    results = {}
    for task in tasks:
        file_path = os.path.join(predictions_dir, f"{task}{suffix}")
        if not os.path.isfile(file_path):
            print(f"  [SKIP] Prediction file not found: {file_path}")
            continue

        scores = []
        # Determine format by extension
        if file_path.endswith(".jsonl"):
            import jsonlines
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    if answer_key not in obj or prediction_key not in obj:
                        raise KeyError(
                            f"Missing '{answer_key}' or '{prediction_key}' in predictions for {task}."
                        )
                    scores.append(exact_match(str(obj[prediction_key]), str(obj[answer_key])))
        else:
            # JSON array of objects
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    if answer_key not in obj or prediction_key not in obj:
                        raise KeyError(
                            f"Missing '{answer_key}' or '{prediction_key}' in predictions for {task}."
                        )
                    scores.append(exact_match(str(obj[prediction_key]), str(obj[answer_key])))
            else:
                raise ValueError(f"Unexpected format in {file_path}")

        results[task] = calculate_accuracy(scores)
    return results


def print_matrix(tasks: List[str], matrix: List[List[Optional[float]]]) -> None:
    """Pretty-print the accuracy matrix."""
    header = "            " + "  ".join(f"{t:>12s}" for t in tasks)
    print(header)
    for i, row in enumerate(matrix):
        row_str = f"  Stage {i+1}:  "
        for val in row:
            if val is not None:
                row_str += f"  {val:12.4f}"
            else:
                row_str += f"  {'--':>12s}"
        print(row_str)


def print_continual_report(
    tasks: List[str],
    matrix: List[List[Optional[float]]],
    stage: int,
) -> Dict[str, float]:
    """Print and return continual learning metrics for a given stage."""
    metrics: Dict[str, float] = {}

    # Stage-level metrics
    acc_stage, fgt_stage = compute_stage_acc_fgt(matrix, stage)
    metrics["ACC_stage"] = acc_stage
    metrics["Fgt_stage"] = fgt_stage

    # If we finished all stages, compute global ACC / Fgt
    if stage == len(tasks) and all(v is not None for v in matrix[-1]):
        acc, fgt = compute_acc_fgt(matrix)
        metrics["ACC"] = acc
        metrics["Fgt"] = fgt

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual learning evaluation tracker.")
    parser.add_argument("--tasks", required=True, help="Comma-separated task list in order.")
    parser.add_argument("--stage", type=int, required=True, help="1-based stage index after training task t.")
    parser.add_argument("--matrix_path", required=True, help="Path to save the accuracy matrix JSON.")
    parser.add_argument(
        "--results_json",
        default="",
        help="Optional path to a JSON mapping task name to accuracy.",
    )
    parser.add_argument(
        "--predictions_dir",
        default="",
        help="Optional directory with per-task prediction files (JSONL or JSON).",
    )
    parser.add_argument(
        "--predictions_suffix",
        default=".jsonl",
        help="Suffix for prediction files (default: .jsonl).",
    )
    parser.add_argument("--answer_key", default="answer", help="Answer key in predictions.")
    parser.add_argument(
        "--prediction_key",
        default="generated_text",
        help="Prediction key in predictions.",
    )
    parser.add_argument(
        "--metrics_path",
        default="",
        help="Optional path to save ACC/Fgt metrics as JSON.",
    )

    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    if not tasks:
        raise ValueError("Task list must not be empty.")

    if args.results_json:
        results = load_results_json(args.results_json)
    elif args.predictions_dir:
        results = compute_exact_match_from_predictions(
            args.predictions_dir,
            tasks,
            args.answer_key,
            args.prediction_key,
            args.predictions_suffix,
        )
    else:
        raise ValueError("Provide either --results_json or --predictions_dir.")

    matrix_data = update_accuracy_matrix(args.matrix_path, args.stage, tasks, results)
    matrix = matrix_data["matrix"]

    # Print accuracy matrix
    print("\nAccuracy matrix:")
    print_matrix(tasks, matrix)

    # Compute and display metrics
    metrics = print_continual_report(tasks, matrix, args.stage)

    print(f"\nStage {args.stage} metrics: {metrics}")

    if args.metrics_path:
        save_json(args.metrics_path, metrics)
        print(f"Metrics saved to {args.metrics_path}")


if __name__ == "__main__":
    main()
