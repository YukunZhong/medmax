import argparse
import os
from typing import Dict, List

import jsonlines

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
    results = {}
    for task in tasks:
        file_path = os.path.join(predictions_dir, f"{task}{suffix}")
        scores = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                if answer_key not in obj or prediction_key not in obj:
                    raise KeyError(
                        f"Missing '{answer_key}' or '{prediction_key}' in predictions for {task}."
                    )
                scores.append(exact_match(str(obj[prediction_key]), str(obj[answer_key])))
        results[task] = calculate_accuracy(scores)
    return results


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
        help="Optional directory with per-task prediction JSONL files.",
    )
    parser.add_argument(
        "--predictions_suffix",
        default=".jsonl",
        help="Suffix for prediction files (default: .jsonl).",
    )
    parser.add_argument("--answer_key", default="answer", help="Answer key in predictions JSONL.")
    parser.add_argument(
        "--prediction_key",
        default="generated_text",
        help="Prediction key in predictions JSONL.",
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

    metrics = {}
    if args.stage == len(tasks) and all(value is not None for value in matrix[-1]):
        acc, fgt = compute_acc_fgt(matrix)
        metrics["ACC"] = acc
        metrics["Fgt"] = fgt
    acc_stage, fgt_stage = compute_stage_acc_fgt(matrix, args.stage)
    metrics["ACC_stage"] = acc_stage
    metrics["Fgt_stage"] = fgt_stage

    if args.metrics_path:
        save_json(args.metrics_path, metrics)
    else:
        print(metrics)


if __name__ == "__main__":
    main()
