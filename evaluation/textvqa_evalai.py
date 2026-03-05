#!/usr/bin/env python3
"""
TextVQA EvalAI 评估脚本 – 通过 EvalAI API 提交 test set 预测结果获取准确率。

TextVQA test set 没有公开的 ground truth，只能通过 EvalAI 在线评估。

─────────────────────────────────────────────────
使用方式
─────────────────────────────────────────────────

1. 获取 EvalAI Token:
   https://eval.ai/ → 登录 → Profile → Auth Token

2. 提交预测结果:
   python -m evaluation.textvqa_evalai submit \
       --submission_file  output_dir/continual_eval_results/stage2_predictions/textvqa_evalai_submission.json \
       --evalai_token     YOUR_EVALAI_AUTH_TOKEN

   加 --wait 可以自动轮询等待评估完成并打印准确率:
   python -m evaluation.textvqa_evalai submit \
       --submission_file  output_dir/continual_eval_results/stage2_predictions/textvqa_evalai_submission.json \
       --evalai_token     YOUR_EVALAI_AUTH_TOKEN \
       --wait

3. 查询已提交任务的状态:
   python -m evaluation.textvqa_evalai status \
       --submission_id  12345 \
       --evalai_token   YOUR_EVALAI_AUTH_TOKEN

4. 格式转换 (内部预测格式 → EvalAI 提交格式):
   python -m evaluation.textvqa_evalai convert \
       --input_file   raw_predictions.json \
       --output_file  evalai_submission.json

5. 批量提交多个文件:
   python -m evaluation.textvqa_evalai batch_submit \
       --submission_dir  output_dir/continual_eval_results/ \
       --evalai_token    YOUR_EVALAI_AUTH_TOKEN \
       --wait

提交文件格式: [{"question_id": int, "answer": str}, ...]
Challenge ID 874 = TextVQA 0.5.1 官方 challenge
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# EvalAI 配置
# ──────────────────────────────────────────────────────────────────────

EVALAI_API_BASE = "https://eval.ai/api"
TEXTVQA_CHALLENGE_ID = 874  # TextVQA 0.5.1 官方 challenge


# ──────────────────────────────────────────────────────────────────────
# 提交文件格式校验
# ──────────────────────────────────────────────────────────────────────

def validate_submission_file(filepath: str) -> List[Dict]:
    """
    校验提交文件格式是否符合 EvalAI 要求。

    合法格式: [{"question_id": int, "answer": str}, ...]
    返回解析后的列表，不合法则报错退出。
    """
    if not os.path.isfile(filepath):
        print(f"错误: 文件不存在 – {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"错误: JSON 解析失败 – {e}")
            sys.exit(1)

    if not isinstance(data, list):
        print(f"错误: 提交文件顶层应为 list，实际为 {type(data).__name__}")
        sys.exit(1)

    if len(data) == 0:
        print("错误: 提交文件为空列表")
        sys.exit(1)

    # 校验每条记录
    errors = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"  第 {i} 条: 应为 dict，实际为 {type(item).__name__}")
            continue
        if "question_id" not in item:
            errors.append(f"  第 {i} 条: 缺少 'question_id' 字段")
        if "answer" not in item:
            errors.append(f"  第 {i} 条: 缺少 'answer' 字段")
        if len(errors) >= 10:
            errors.append("  ... (更多错误省略)")
            break

    if errors:
        print("提交文件格式错误:")
        for e in errors:
            print(e)
        sys.exit(1)

    # 检查 question_id 唯一性
    qids = [item["question_id"] for item in data]
    if len(qids) != len(set(qids)):
        dup_count = len(qids) - len(set(qids))
        print(f"警告: 有 {dup_count} 个重复的 question_id")

    print(f"文件格式校验通过: {len(data)} 条预测结果")
    return data


# ──────────────────────────────────────────────────────────────────────
# EvalAI API 提交
# ──────────────────────────────────────────────────────────────────────

def _get_requests():
    """导入并返回 requests 模块。"""
    try:
        import requests
        return requests
    except ImportError:
        print("错误: 请先安装 requests 库")
        print("  pip install requests")
        sys.exit(1)


def evalai_get_phases(
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
) -> List[Dict]:
    """获取 EvalAI challenge 的所有 phases。"""
    requests = _get_requests()
    headers = {"Authorization": f"Bearer {auth_token}"}

    print(f"正在获取 Challenge {challenge_id} 的 phases...")
    resp = requests.get(
        f"{EVALAI_API_BASE}/challenges/challenge/{challenge_id}/challenge_phase",
        headers=headers,
    )

    if resp.status_code != 200:
        print(f"获取 phases 失败: HTTP {resp.status_code}")
        print(f"  响应: {resp.text[:500]}")
        sys.exit(1)

    phases = resp.json()
    if isinstance(phases, dict) and "results" in phases:
        phases = phases["results"]

    return phases


def evalai_resolve_phase_id(
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
    phase_id: Optional[int] = None,
) -> int:
    """
    解析 phase_id。如果用户没有指定，自动从 challenge 中获取 test phase。
    """
    if phase_id is not None:
        return phase_id

    phases = evalai_get_phases(auth_token, challenge_id)

    if not phases:
        print("错误: 未找到可用的 challenge phase")
        sys.exit(1)

    print(f"  找到 {len(phases)} 个 phases:")
    for p in phases:
        print(f"    Phase {p['id']}: {p.get('name', 'N/A')} "
              f"(codename: {p.get('codename', 'N/A')})")

    # 优先选择包含 "test" 的 phase
    test_phases = [
        p for p in phases
        if "test" in p.get("name", "").lower()
        or "test" in p.get("codename", "").lower()
    ]
    if test_phases:
        chosen = test_phases[0]
    else:
        chosen = phases[-1]

    print(f"  → 自动选择 phase_id: {chosen['id']} ({chosen.get('name', 'N/A')})")
    return chosen["id"]


def _ensure_participation(
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
) -> None:
    """
    确保用户已报名参加 challenge。
    如果尚未参加，自动创建参赛队伍并报名。
    """
    requests = _get_requests()
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Step 1: 检查是否已有 participant team
    print("检查参赛状态...")
    resp = requests.get(
        f"{EVALAI_API_BASE}/participants/participant_team",
        headers=headers,
    )
    if resp.status_code != 200:
        print(f"  获取参赛队伍失败: HTTP {resp.status_code}")
        print(f"  响应: {resp.text[:300]}")
        return

    teams_data = resp.json()
    teams = teams_data.get("results", teams_data) if isinstance(teams_data, dict) else teams_data
    if not isinstance(teams, list):
        teams = []

    team_id = None

    if teams:
        # 用已有的第一个队伍
        team_id = teams[0]["id"]
        team_name = teams[0].get("team_name", "N/A")
        print(f"  已有参赛队伍: {team_name} (id={team_id})")
    else:
        # Step 2: 创建一个新队伍
        import getpass
        import socket
        default_name = f"team_{getpass.getuser()}_{socket.gethostname()}"
        print(f"  未找到参赛队伍，正在创建: {default_name}")
        resp = requests.post(
            f"{EVALAI_API_BASE}/participants/participant_team",
            headers={**headers, "Content-Type": "application/json"},
            json={"team_name": default_name},
        )
        if resp.status_code in (200, 201):
            team_data = resp.json()
            team_id = team_data["id"]
            print(f"  ✓ 创建队伍成功: {default_name} (id={team_id})")
        else:
            print(f"  创建队伍失败: HTTP {resp.status_code}")
            print(f"  响应: {resp.text[:300]}")
            print("  请手动在 https://eval.ai 上创建队伍并报名 challenge")
            return

    # Step 3: 尝试报名 challenge
    print(f"  正在报名 Challenge {challenge_id}...")
    resp = requests.post(
        f"{EVALAI_API_BASE}/participants/participate_in_challenge/{challenge_id}/team/{team_id}",
        headers=headers,
    )
    if resp.status_code in (200, 201):
        print(f"  ✓ 报名成功!")
    elif resp.status_code == 200 or "already" in resp.text.lower():
        print(f"  已参加该 challenge")
    else:
        # 有些情况返回 4xx 但实际已经报名了
        resp_text = resp.text.lower()
        if "already" in resp_text or "exist" in resp_text:
            print(f"  已参加该 challenge")
        else:
            print(f"  报名响应: HTTP {resp.status_code} – {resp.text[:300]}")
            print("  如报名失败，请手动访问:")
            print(f"    https://eval.ai/web/challenges/challenge-page/{challenge_id}/overview")


def evalai_submit(
    submission_file: str,
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
    phase_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    通过 EvalAI REST API 提交 TextVQA test set 预测结果。

    参数:
      submission_file: EvalAI 格式的 JSON 文件路径
                       [{"question_id": int, "answer": str}, ...]
      auth_token:  EvalAI 用户认证 Token
      challenge_id: EvalAI challenge ID (默认 874 = TextVQA 0.5.1)
      phase_id:    Challenge phase ID (不提供则自动获取)

    返回: API 响应 dict（含 submission id 等信息）
    """
    requests = _get_requests()

    # 校验文件
    validate_submission_file(submission_file)

    # 确保已报名参加 challenge
    _ensure_participation(auth_token, challenge_id)

    # 解析 phase_id
    phase_id = evalai_resolve_phase_id(auth_token, challenge_id, phase_id)

    # 提交
    headers = {"Authorization": f"Bearer {auth_token}"}

    print(f"\n正在提交到 EvalAI...")
    print(f"  Challenge: {challenge_id}")
    print(f"  Phase:     {phase_id}")
    print(f"  File:      {submission_file}")

    with open(submission_file, "rb") as f:
        resp = requests.post(
            f"{EVALAI_API_BASE}/jobs/challenge/{challenge_id}"
            f"/challenge_phase/{phase_id}/submission/",
            headers=headers,
            files={"input_file": (os.path.basename(submission_file), f, "application/json")},
            data={"status": "submitting"},
        )

    if resp.status_code in (200, 201):
        result = resp.json()
        submission_id = result.get("id", "unknown")
        print(f"\n✓ 提交成功!")
        print(f"  Submission ID: {submission_id}")
        print(f"  状态: {result.get('status', 'unknown')}")
        print(f"  查看结果: https://eval.ai/web/challenges/"
              f"challenge-page/{challenge_id}/my-submissions")
        return result
    else:
        print(f"\n✗ 提交失败: HTTP {resp.status_code}")
        print(f"  响应: {resp.text[:1000]}")
        return {"error": resp.text, "status_code": resp.status_code}


# ──────────────────────────────────────────────────────────────────────
# 查询提交状态 / 获取结果
# ──────────────────────────────────────────────────────────────────────

def evalai_check_status(
    submission_id: int,
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
) -> Dict[str, Any]:
    """查询 EvalAI 提交状态及评估结果。"""
    requests = _get_requests()

    headers = {"Authorization": f"Bearer {auth_token}"}
    resp = requests.get(
        f"{EVALAI_API_BASE}/jobs/submission/{submission_id}",
        headers=headers,
    )

    if resp.status_code != 200:
        print(f"查询失败: HTTP {resp.status_code}")
        print(f"  响应: {resp.text[:500]}")
        return {"error": resp.text}

    result = resp.json()
    status = result.get("status", "unknown")
    print(f"Submission {submission_id} 状态: {status}")

    if status == "finished":
        _print_eval_results(result)
    elif status == "running":
        print("  评估正在进行中，请稍后再查询...")
    elif status == "submitted":
        print("  已提交，等待排队评估...")
    elif status == "failed":
        stderr = result.get("stderr_file", "")
        print(f"  评估失败!")
        if stderr:
            print(f"  错误日志: {stderr}")

    return result


def _print_eval_results(result: Dict[str, Any]):
    """打印 EvalAI 评估结果。"""
    result_data = result.get("result", [])

    print("\n" + "=" * 60)
    print("  TextVQA EvalAI 评估结果")
    print("=" * 60)

    if isinstance(result_data, list):
        for split_result in result_data:
            if isinstance(split_result, dict):
                for metric_name, metric_value in split_result.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.4f} "
                              f"({metric_value * 100:.2f}%)")
                    else:
                        print(f"  {metric_name}: {metric_value}")
    elif isinstance(result_data, dict):
        for metric_name, metric_value in result_data.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f} "
                      f"({metric_value * 100:.2f}%)")
            else:
                print(f"  {metric_name}: {metric_value}")
    else:
        print(f"  {result_data}")

    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# 提交并等待结果
# ──────────────────────────────────────────────────────────────────────

def evalai_submit_and_wait(
    submission_file: str,
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
    phase_id: Optional[int] = None,
    poll_interval: int = 30,
    max_wait: int = 600,
) -> Dict[str, Any]:
    """
    提交到 EvalAI 并自动轮询等待评估结果。

    参数:
      poll_interval: 轮询间隔秒数 (默认 30)
      max_wait:      最大等待时间秒数 (默认 600)
    """
    # 提交
    submit_result = evalai_submit(
        submission_file, auth_token, challenge_id, phase_id
    )
    if "error" in submit_result:
        return submit_result

    submission_id = submit_result.get("id")
    if not submission_id:
        print("无法获取 submission ID，请手动查询")
        return submit_result

    # 轮询
    print(f"\n等待评估完成 (每 {poll_interval}s 轮询, 最长 {max_wait}s)...")
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        status_result = evalai_check_status(
            submission_id, auth_token, challenge_id
        )
        status = status_result.get("status", "unknown")

        if status == "finished":
            return status_result
        elif status == "failed":
            return status_result

        print(f"  [{elapsed}s/{max_wait}s] 状态: {status}, 继续等待...")

    print(f"\n等待超时 ({max_wait}s)，可稍后用 status 命令查询:")
    print(f"  python -m evaluation.textvqa_evalai status "
          f"--submission_id {submission_id} --evalai_token YOUR_TOKEN")
    return {"status": "timeout", "submission_id": submission_id}


# ──────────────────────────────────────────────────────────────────────
# 格式转换: 内部预测格式 → EvalAI 提交格式
# ──────────────────────────────────────────────────────────────────────

def convert_to_evalai_format(input_file: str, output_file: str) -> str:
    """
    将内部预测格式转换为 EvalAI 提交格式。

    支持的输入格式:
      1. [{"question_id": int, "answer": str}, ...]           → 已经是 EvalAI 格式
      2. [{"question_id": int, "prediction": str, ...}, ...]  → 内部格式
      3. {"predictions": [...]}                                → 包裹格式

    输出格式:
      [{"question_id": int, "answer": str}, ...]
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取预测列表
    if isinstance(data, dict) and "predictions" in data:
        preds = data["predictions"]
    elif isinstance(data, list):
        preds = data
    else:
        print(f"错误: 无法识别输入文件格式")
        sys.exit(1)

    # 检查是否已经是 EvalAI 格式
    if preds and "answer" in preds[0] and "prediction" not in preds[0]:
        print(f"文件已经是 EvalAI 提交格式: {input_file}")
        if input_file != output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2)
            print(f"已复制到: {output_file}")
        return output_file

    # 转换
    submission = []
    for p in preds:
        qid = p.get("question_id")
        answer = p.get("prediction", p.get("answer", ""))
        if qid is None:
            continue
        submission.append({"question_id": qid, "answer": str(answer)})

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print(f"已转换为 EvalAI 格式: {output_file} ({len(submission)} 条)")
    return output_file


# ──────────────────────────────────────────────────────────────────────
# 批量提交
# ──────────────────────────────────────────────────────────────────────

def batch_submit(
    submission_dir: str,
    auth_token: str,
    challenge_id: int = TEXTVQA_CHALLENGE_ID,
    phase_id: Optional[int] = None,
    wait: bool = False,
    poll_interval: int = 30,
    max_wait: int = 600,
    file_pattern: str = "**/textvqa_evalai_submission.json",
) -> List[Dict[str, Any]]:
    """
    批量提交目录下所有匹配的 submission 文件。
    """
    pattern = os.path.join(submission_dir, file_pattern)
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"未找到匹配的提交文件: {pattern}")
        return []

    print(f"找到 {len(files)} 个提交文件:")
    for f in files:
        print(f"  {f}")
    print()

    # 预先解析 phase_id（避免每次都查询）
    phase_id = evalai_resolve_phase_id(auth_token, challenge_id, phase_id)

    results = []
    for i, filepath in enumerate(files, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(files)}] 提交: {filepath}")
        print(f"{'─' * 60}")

        if wait:
            result = evalai_submit_and_wait(
                filepath, auth_token, challenge_id, phase_id,
                poll_interval=poll_interval, max_wait=max_wait,
            )
        else:
            result = evalai_submit(
                filepath, auth_token, challenge_id, phase_id,
            )

        result["_file"] = filepath
        results.append(result)

        # 批量提交间隔，避免速率限制
        if i < len(files):
            print("  等待 5s 后提交下一个...")
            time.sleep(5)

    # 打印汇总
    print(f"\n{'=' * 60}")
    print(f"批量提交完成 ({len(results)}/{len(files)})")
    print(f"{'=' * 60}")
    for r in results:
        fname = os.path.basename(os.path.dirname(r.get("_file", "")))
        sid = r.get("id", "N/A")
        st = r.get("status", r.get("error", "unknown")[:50])
        print(f"  {fname}: submission_id={sid}, status={st}")

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TextVQA EvalAI 评估工具 – 提交 test set 预测结果到 EvalAI 平台获取准确率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提交并等待结果
  python -m evaluation.textvqa_evalai submit \\
      --submission_file textvqa_evalai_submission.json \\
      --evalai_token YOUR_TOKEN --wait

  # 查询已有提交的状态
  python -m evaluation.textvqa_evalai status \\
      --submission_id 12345 --evalai_token YOUR_TOKEN

  # 格式转换
  python -m evaluation.textvqa_evalai convert \\
      --input_file raw_preds.json --output_file submission.json

  # 批量提交
  python -m evaluation.textvqa_evalai batch_submit \\
      --submission_dir output_dir/continual_eval_results/ \\
      --evalai_token YOUR_TOKEN --wait
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="操作模式")

    # ─── submit: 提交到 EvalAI ─────────────────────────────────
    sub_submit = subparsers.add_parser(
        "submit",
        help="提交预测结果到 EvalAI 平台",
    )
    sub_submit.add_argument(
        "--submission_file", type=str, required=True,
        help='EvalAI 格式的 JSON 文件: [{"question_id": int, "answer": str}, ...]'
    )
    sub_submit.add_argument(
        "--evalai_token", type=str, required=True,
        help="EvalAI 认证 Token (从 https://eval.ai → Profile → Auth Token 获取)"
    )
    sub_submit.add_argument(
        "--challenge_id", type=int, default=TEXTVQA_CHALLENGE_ID,
        help=f"EvalAI Challenge ID (默认: {TEXTVQA_CHALLENGE_ID})"
    )
    sub_submit.add_argument(
        "--phase_id", type=int, default=None,
        help="Challenge Phase ID (不提供则自动获取 test phase)"
    )
    sub_submit.add_argument(
        "--wait", action="store_true",
        help="提交后自动轮询等待评估完成并打印准确率"
    )
    sub_submit.add_argument(
        "--poll_interval", type=int, default=30,
        help="轮询间隔秒数 (默认: 30)"
    )
    sub_submit.add_argument(
        "--max_wait", type=int, default=600,
        help="最大等待时间秒数 (默认: 600)"
    )

    # ─── status: 查询提交状态 ──────────────────────────────────
    sub_status = subparsers.add_parser(
        "status",
        help="查询已提交任务的状态和评估结果",
    )
    sub_status.add_argument(
        "--submission_id", type=int, required=True,
        help="EvalAI Submission ID"
    )
    sub_status.add_argument(
        "--evalai_token", type=str, required=True,
        help="EvalAI 认证 Token"
    )
    sub_status.add_argument(
        "--challenge_id", type=int, default=TEXTVQA_CHALLENGE_ID,
    )

    # ─── convert: 格式转换 ──────────────────────────────────────
    sub_convert = subparsers.add_parser(
        "convert",
        help="将内部预测格式转换为 EvalAI 提交格式",
    )
    sub_convert.add_argument(
        "--input_file", type=str, required=True,
        help="输入文件（内部预测格式）"
    )
    sub_convert.add_argument(
        "--output_file", type=str, required=True,
        help="输出文件（EvalAI 提交格式）"
    )

    # ─── batch_submit: 批量提交 ──────────────────────────────────
    sub_batch = subparsers.add_parser(
        "batch_submit",
        help="批量提交目录下所有 textvqa_evalai_submission.json",
    )
    sub_batch.add_argument(
        "--submission_dir", type=str, required=True,
        help="包含提交文件的根目录（递归搜索）"
    )
    sub_batch.add_argument(
        "--evalai_token", type=str, required=True,
        help="EvalAI 认证 Token"
    )
    sub_batch.add_argument(
        "--challenge_id", type=int, default=TEXTVQA_CHALLENGE_ID,
    )
    sub_batch.add_argument(
        "--phase_id", type=int, default=None,
    )
    sub_batch.add_argument(
        "--file_pattern", type=str, default="**/textvqa_evalai_submission.json",
        help="提交文件 glob 模式 (默认: **/textvqa_evalai_submission.json)"
    )
    sub_batch.add_argument(
        "--wait", action="store_true",
        help="每个提交都等待评估完成"
    )
    sub_batch.add_argument(
        "--poll_interval", type=int, default=30,
    )
    sub_batch.add_argument(
        "--max_wait", type=int, default=600,
    )

    args = parser.parse_args()

    if args.command == "submit":
        if args.wait:
            evalai_submit_and_wait(
                args.submission_file,
                args.evalai_token,
                args.challenge_id,
                args.phase_id,
                poll_interval=args.poll_interval,
                max_wait=args.max_wait,
            )
        else:
            evalai_submit(
                args.submission_file,
                args.evalai_token,
                args.challenge_id,
                args.phase_id,
            )

    elif args.command == "status":
        evalai_check_status(
            args.submission_id,
            args.evalai_token,
            args.challenge_id,
        )

    elif args.command == "convert":
        convert_to_evalai_format(args.input_file, args.output_file)

    elif args.command == "batch_submit":
        batch_submit(
            args.submission_dir,
            args.evalai_token,
            args.challenge_id,
            args.phase_id,
            wait=args.wait,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
            file_pattern=args.file_pattern,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
