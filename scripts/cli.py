#!/usr/bin/env python3
"""MedAgent 命令行入口

    python scripts/cli.py version
    python scripts/cli.py cases
    python scripts/cli.py run --case-id mock-mi-001 --skill 0.9
    python scripts/cli.py batch --n 6
    python scripts/cli.py serve --port 8000
    python scripts/cli.py build-swift-data --mode gym
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.medagent.demo_agent import run_episode
from src.medagent.env import CaseCatalog, MedAgentEnv
from src.medagent.version import __version__


def cmd_version(_: argparse.Namespace) -> int:
    print(f"MedAgent {__version__}")
    return 0


def cmd_cases(args: argparse.Namespace) -> int:
    catalog = CaseCatalog()
    for item in catalog.list():
        print(f"{item['case_id']:<14} {item['department']:<8} {item['difficulty']:<7} {item['chief_complaint']}")
    print(f"\n共 {len(catalog.cases)} 个病例（{catalog.path}）")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    env = MedAgentEnv(env_config={"case_id": args.case_id, "max_steps": args.max_steps})
    result = asyncio.run(run_episode(env, skill=args.skill, max_turns=args.max_steps))

    print("=" * 62)
    print(f"病例 {result['case']['case_id']}（{result['case']['department']}）")
    print(f"主诉：{result['case']['chief_complaint']}")
    print("=" * 62)
    for message in result["transcript"]:
        role = message["role"]
        content = message["content"]
        if role == "system" and not message.get("reward"):
            continue
        if role == "assistant":
            reward = message.get("reward")
            suffix = f"  [{reward:+.3f}]" if reward is not None else ""
            print(f"\n[Agent]{suffix}\n{content}")
        elif role == "user":
            print(f"\n[Env]\n{content}")

    print("\n" + "-" * 62)
    print(f"诊断: {result['diagnosis'] or '未给出'}")
    print(f"处置: {result['treatment'] or '未给出'}")
    print(f"得分: {result['score']:.2f}  轮次: {result['turns']}  成本: {result['cost']:.0f} 元  "
          f"累计奖励: {result['total_reward']:.2f}")
    print(f"拆解: {json.dumps(result['breakdown'], ensure_ascii=False)}")
    print(f"标准答案: {result['ground_truth']['diagnosis']} / {result['ground_truth']['treatment']}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    catalog = CaseCatalog()
    cases = catalog.list()[: args.n]
    if not cases:
        print("没有病例", file=sys.stderr)
        return 1

    rows = []
    for item in cases:
        env = MedAgentEnv(env_config={"case_id": item["case_id"], "max_steps": args.max_steps})
        result = asyncio.run(run_episode(env, skill=args.skill, max_turns=args.max_steps))
        rows.append(result)

    print(f"{'case_id':<14}{'科室':<10}{'轮次':>4}{'成本':>8}{'得分':>8}{'奖励':>8}")
    print("-" * 56)
    for result in rows:
        print(f"{result['case']['case_id']:<14}{result['case']['department']:<10}"
              f"{result['turns']:>4}{result['cost']:>8.0f}{result['score']:>8.2f}{result['total_reward']:>8.2f}")
    avg_score = sum(result["score"] for result in rows) / len(rows)
    avg_cost = sum(result["cost"] for result in rows) / len(rows)
    print("-" * 56)
    print(f"平均得分 {avg_score:.3f} | 平均成本 {avg_cost:.0f} 元 | n={len(rows)}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """按医生水平扫一遍全部病例，输出可直接投屏/贴进幻灯片的表格"""
    from src.medagent.demo_agent import skill_tier

    catalog = CaseCatalog()
    cases = catalog.list()[: args.n]
    skills = [float(item) for item in str(args.skills).split(",") if item.strip()]
    if not cases or not skills:
        print("没有病例或水平取值", file=sys.stderr)
        return 1

    rows = []
    for skill in skills:
        results = []
        for item in cases:
            env = MedAgentEnv(env_config={"case_id": item["case_id"], "max_steps": args.max_steps})
            results.append(asyncio.run(run_episode(env, skill=skill, max_turns=args.max_steps)))
        rows.append((skill, results))

    if args.detail:
        for skill, results in rows:
            print(f"\n### skill={skill:.2f}（{skill_tier(skill)}）\n")
            print(f"| 病例 | 科室 | 轮次 | 成本(元) | 诊断 | 治疗 | 安全 | 效率 | 总分 |")
            print("|---|---|---|---|---|---|---|---|---|")
            for result in results:
                parts = result["breakdown"]
                print(
                    f"| {result['case']['case_id']} | {result['case']['department']} | {result['turns']} | "
                    f"{result['cost']:.0f} | {parts.get('diagnosis', 0):.2f} | {parts.get('treatment', 0):.2f} | "
                    f"{parts.get('safety', 0):.2f} | {parts.get('efficiency', 0):.2f} | {result['score']:.2f} |"
                )

    print("\n### 按医生水平汇总\n")
    print("| skill | 档位 | 平均总分 | 平均成本(元) | 平均轮次 | 诊断命中 | 治疗覆盖 |")
    print("|---|---|---|---|---|---|---|")
    for skill, results in rows:
        count = len(results)
        avg_score = sum(item["score"] for item in results) / count
        avg_cost = sum(item["cost"] for item in results) / count
        avg_turns = sum(item["turns"] for item in results) / count
        avg_diag = sum(item["breakdown"].get("diagnosis", 0) for item in results) / count
        avg_treat = sum(item["breakdown"].get("treatment", 0) for item in results) / count
        print(f"| {skill:.2f} | {skill_tier(skill)} | {avg_score:.3f} | {avg_cost:.0f} | "
              f"{avg_turns:.1f} | {avg_diag:.2f} | {avg_treat:.2f} |")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from src.medagent.webui import serve

    serve(args.host, args.port)
    return 0


def cmd_build_swift(args: argparse.Namespace) -> int:
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "build_swift_dataset.py"
    spec = importlib.util.spec_from_file_location("build_swift_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = ["--mode", args.mode, "--max-steps", str(args.max_steps)]
    if args.output:
        argv += ["--output", args.output]
    if args.input:
        argv += ["--input", args.input]
    sys.argv = ["build_swift_dataset.py"] + argv
    return module.main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medagent", description="MedAgent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="打印版本").set_defaults(func=cmd_version)

    cases = sub.add_parser("cases", help="列出病例")
    cases.set_defaults(func=cmd_cases)

    run = sub.add_parser("run", help="运行单个病例")
    run.add_argument("--case-id", default=None)
    run.add_argument("--skill", type=float, default=0.9, help="医生水平 0.2-1.0")
    run.add_argument("--max-steps", type=int, default=16)
    run.set_defaults(func=cmd_run)

    batch = sub.add_parser("batch", help="批量运行 mock 病例")
    batch.add_argument("--n", type=int, default=6)
    batch.add_argument("--skill", type=float, default=0.9)
    batch.add_argument("--max-steps", type=int, default=16)
    batch.set_defaults(func=cmd_batch)

    sweep = sub.add_parser("sweep", help="按医生水平扫全部病例，输出对比表")
    sweep.add_argument("--skills", default="0.9,0.6,0.2", help="逗号分隔的水平取值")
    sweep.add_argument("--n", type=int, default=6)
    sweep.add_argument("--max-steps", type=int, default=16)
    sweep.add_argument("--detail", action="store_true", help="附带每个病例的明细表")
    sweep.set_defaults(func=cmd_sweep)

    serve_cmd = sub.add_parser("serve", help="启动 WebUI + 会话 API")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.set_defaults(func=cmd_serve)

    swift = sub.add_parser("build-swift-data", help="构造 ms-swift 数据集")
    swift.add_argument("--mode", choices=["gym", "prompt"], default="gym")
    swift.add_argument("--input", default=None)
    swift.add_argument("--output", default=None)
    swift.add_argument("--max-steps", type=int, default=16)
    swift.set_defaults(func=cmd_build_swift)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())