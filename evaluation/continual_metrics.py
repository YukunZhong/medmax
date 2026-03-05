"""
Continual learning metrics for an N-task accuracy matrix.

The accuracy matrix ``M`` has shape ``(N, N)`` where ``M[i][j]`` is the
accuracy on task ``j`` measured after training stage ``i`` (0-indexed).
Only entries where ``j <= i`` are expected to be filled.

Metrics
-------
ACC  – Average accuracy on *all* tasks after the final training stage.
       ``ACC = mean(M[N-1][0..N-1])``

Fgt  – Average forgetting across tasks 0..N-2.
       For task j: ``fgt_j = max_{t>=j}(M[t][j]) - M[N-1][j]``
       ``Fgt = mean(fgt_0, ..., fgt_{N-2})``

BWT  – Backward transfer.
       ``BWT = (1/(N-1)) * sum_{j=0}^{N-2} (M[N-1][j] - M[j][j])``
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, reference: str) -> int:
    """Return 1 if normalised prediction equals normalised reference."""
    return int(normalize_answer(prediction) == normalize_answer(reference))


def calculate_accuracy(scores: List[int]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ── JSON I/O ──────────────────────────────────────────────────────

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict) -> None:
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Accuracy matrix management ───────────────────────────────────

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


# ── Continual learning metrics ───────────────────────────────────

def compute_acc_fgt(matrix: List[List[Optional[float]]]) -> Tuple[float, float]:
    """Compute final-stage ACC and average Forgetting over the full matrix.

    None entries (submission-only tasks without ground-truth labels) are
    silently excluded from all averages.
    """
    if not matrix:
        raise ValueError("Empty accuracy matrix.")
    last_row = matrix[-1]
    labelled = [v for v in last_row if v is not None]
    if not labelled:
        return 0.0, 0.0
    acc = sum(labelled) / len(labelled)

    fgt_values = []
    num_tasks = len(matrix)
    for task_idx in range(num_tasks - 1):
        # Skip tasks that have no ground-truth labels.
        if matrix[-1][task_idx] is None:
            continue
        history = [matrix[t][task_idx] for t in range(task_idx, num_tasks) if matrix[t][task_idx] is not None]
        if not history:
            continue
        fgt_values.append(max(history) - matrix[-1][task_idx])
    fgt = sum(fgt_values) / len(fgt_values) if fgt_values else 0.0
    return acc, fgt


def compute_bwt(matrix: List[List[Optional[float]]]) -> float:
    """Compute Backward Transfer (BWT).

    BWT = (1/(N-1)) * sum_{j=0}^{N-2} (M[N-1][j] - M[j][j])
    Positive BWT means later training *improved* earlier tasks.
    Negative BWT indicates forgetting.
    None entries (submission-only tasks) are excluded.
    """
    num_tasks = len(matrix)
    if num_tasks < 2:
        return 0.0
    last_row = matrix[-1]
    bwt_values = []
    for j in range(num_tasks - 1):
        # Skip submission-only tasks that have no labels.
        if matrix[j][j] is None or last_row[j] is None:
            continue
        bwt_values.append(last_row[j] - matrix[j][j])
    return sum(bwt_values) / len(bwt_values) if bwt_values else 0.0


def compute_stage_acc_fgt(matrix: List[List[Optional[float]]], stage: int) -> Tuple[float, float]:
    """Compute ACC and Fgt for a particular stage (1-based).

    None entries (submission-only tasks without ground-truth labels) are
    excluded from averages.
    """
    if stage < 1 or stage > len(matrix):
        raise ValueError("Stage out of bounds for matrix.")
    row_idx = stage - 1
    row = matrix[row_idx]
    labelled = [v for v in row[:stage] if v is not None]
    acc = sum(labelled) / len(labelled) if labelled else 0.0

    fgt_values = []
    for task_idx in range(stage - 1):
        # Skip tasks with no labels.
        if matrix[row_idx][task_idx] is None:
            continue
        history = [matrix[t][task_idx] for t in range(task_idx, stage) if matrix[t][task_idx] is not None]
        if not history:
            continue
        fgt_values.append(max(history) - matrix[row_idx][task_idx])
    fgt = sum(fgt_values) / len(fgt_values) if fgt_values else 0.0
    return acc, fgt


def format_matrix(tasks: List[str], matrix: List[List[Optional[float]]]) -> str:
    """Return a pretty-printed string of the accuracy matrix."""
    lines = []
    header = "            " + "  ".join(f"{t:>12s}" for t in tasks)
    lines.append(header)
    for i, row in enumerate(matrix):
        row_str = f"  Stage {i+1}:  "
        for val in row:
            if val is not None:
                row_str += f"  {val:12.4f}"
            else:
                row_str += f"  {'--':>12s}"
        lines.append(row_str)
    return "\n".join(lines)
