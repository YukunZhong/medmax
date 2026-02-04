import json
import os
import re
from typing import Dict, List, Optional, Tuple


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, reference: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(reference))


def calculate_accuracy(scores: List[int]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def init_accuracy_matrix(tasks: List[str]) -> Dict:
    size = len(tasks)
    matrix = [[None for _ in range(size)] for _ in range(size)]
    return {"tasks": tasks, "matrix": matrix}


def load_accuracy_matrix(matrix_path: str, tasks: List[str]) -> Dict:
    if not os.path.exists(matrix_path):
        return init_accuracy_matrix(tasks)
    data = load_json(matrix_path)
    if data.get("tasks") != tasks:
        raise ValueError("Task list mismatch between matrix file and current run.")
    return data


def update_accuracy_matrix(
    matrix_path: str,
    stage: int,
    tasks: List[str],
    results: Dict[str, float],
) -> Dict:
    data = load_accuracy_matrix(matrix_path, tasks)
    matrix = data["matrix"]
    if stage < 1 or stage > len(tasks):
        raise ValueError("Stage must be within the range of tasks.")
    row_idx = stage - 1
    for col_idx, task in enumerate(tasks):
        if task in results:
            matrix[row_idx][col_idx] = results[task]
    data["matrix"] = matrix
    save_json(matrix_path, data)
    return data


def compute_acc_fgt(matrix: List[List[Optional[float]]]) -> Tuple[float, float]:
    if not matrix:
        raise ValueError("Empty accuracy matrix.")
    last_row = matrix[-1]
    if any(value is None for value in last_row):
        raise ValueError("Final row contains missing accuracy values.")
    acc = sum(last_row) / len(last_row)

    fgt_values = []
    num_tasks = len(matrix)
    for task_idx in range(num_tasks - 1):
        history = [matrix[t][task_idx] for t in range(task_idx, num_tasks) if matrix[t][task_idx] is not None]
        if not history:
            continue
        fgt_values.append(max(history) - matrix[-1][task_idx])
    fgt = sum(fgt_values) / len(fgt_values) if fgt_values else 0.0
    return acc, fgt


def compute_stage_acc_fgt(matrix: List[List[Optional[float]]], stage: int) -> Tuple[float, float]:
    if stage < 1 or stage > len(matrix):
        raise ValueError("Stage out of bounds for matrix.")
    row_idx = stage - 1
    row = matrix[row_idx]
    if any(value is None for value in row[: stage]):
        raise ValueError("Stage row has missing accuracy values.")
    acc = sum(row[:stage]) / stage

    fgt_values = []
    for task_idx in range(stage - 1):
        history = [matrix[t][task_idx] for t in range(task_idx, stage) if matrix[t][task_idx] is not None]
        if not history:
            continue
        fgt_values.append(max(history) - matrix[row_idx][task_idx])
    fgt = sum(fgt_values) / len(fgt_values) if fgt_values else 0.0
    return acc, fgt
