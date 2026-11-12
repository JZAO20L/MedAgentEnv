#!/usr/bin/env python3
"""把病例 JSONL 转成 ms-swift 可用的数据集

两种模式：
- ``gym``   : 每行一个 env_config，环境自己生成首轮 observation（配合 --use_gym_env true）
- ``prompt``: 单轮 prompt + 标准答案列，只用 reward_funcs（不接环境）

用法：
    python scripts/build_swift_dataset.py --mode gym --output assets/mock/swift_gym.jsonl
    python scripts/build_swift_dataset.py --mode prompt --output assets/mock/swift_prompt.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.medagent.env import SYSTEM_PROMPT

DEFAULT_INPUT = PROJECT_ROOT / "assets" / "mock" / "cases.jsonl"


def load_cases(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_gym_row(case: dict, max_steps: int) -> dict:
    ground_truth = case.get("ground_truth", {})
    return {
        "messages": [{"role": "user", "content": "<placeholder>"}],
        "env_config": {"name": "medagent", "case_id": case["case_id"], "max_steps": max_steps},
        "case_id": case["case_id"],
        "difficulty": case.get("difficulty", "medium"),
        "department": (case.get("tags") or ["未知"])[0],
        "ground_truth_diagnosis": ground_truth.get("diagnosis", []),
        "ground_truth_treatment": ground_truth.get("treatment", []),
        "ground_truth_avoid": ground_truth.get("avoid", []),
    }


def to_prompt_row(case: dict) -> dict:
    ground_truth = case.get("ground_truth", {})
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【病例】主诉：{case['chief_complaint']}"},
        ],
        "case_id": case["case_id"],
        "difficulty": case.get("difficulty", "medium"),
        "department": (case.get("tags") or ["未知"])[0],
        "ground_truth_diagnosis": ground_truth.get("diagnosis", []),
        "ground_truth_treatment": ground_truth.get("treatment", []),
        "ground_truth_avoid": ground_truth.get("avoid", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构造 ms-swift 数据集")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["gym", "prompt"], default="gym")
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args()

    input_path = Path(args.input)
    cases = load_cases(input_path)
    if not cases:
        print(f"没有病例: {input_path}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else input_path.with_name(f"swift_{args.mode}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for case in cases:
            row = to_gym_row(case, args.max_steps) if args.mode == "gym" else to_prompt_row(case)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已写入 {len(cases)} 行 -> {output}（mode={args.mode}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())