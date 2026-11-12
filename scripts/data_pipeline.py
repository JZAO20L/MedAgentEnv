#!/usr/bin/env python3
"""数据合成 CLI

用法：
    # 质量模式（默认，4 次调用/样本）
    python scripts/data_pipeline.py --seed-data data/seed_cases --output output/synthesis/cases.jsonl

    # 快速模式（2 次调用/样本）
    python scripts/data_pipeline.py --mode fast --max-samples 200

    # 3 个裁判投票（3/3 通过）
    python scripts/data_pipeline.py \
        --judge qwen3.5-plus@https://coding.dashscope.aliyuncs.com/v1@DASHSCOPE_API_KEY_CP \
        --judge glm-5@https://open.bigmodel.cn/api/paas/v4/@GLM_API_KEY \
        --judge baichuan-m2@http://localhost:8200/v1@BAICHUAN_API_KEY

    # 无 API Key 时验证管线
    python scripts/data_pipeline.py --dry-run --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.medagent.data_pipeline import (
    DEFAULT_ROLES,
    PipelineConfig,
    RoleConfig,
    load_seeds,
    make_shards,
    run_pipeline,
)


def parse_role(text: str) -> RoleConfig:
    """解析 model@base_url[@api_key_env]"""
    parts = text.split("@")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("角色格式应为 model@base_url[@api_key_env]")
    model, base_url = parts[0], parts[1]
    api_key_env = parts[2] if len(parts) > 2 else ""
    return RoleConfig(model=model, base_url=base_url, api_key_env=api_key_env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedAgent 数据合成管线")

    data = parser.add_argument_group("数据")
    data.add_argument("--seed-data", default="data/seed_cases", help="种子文件或目录")
    data.add_argument("--output", default="output/synthesis/cases.jsonl", help="通过样本输出")
    data.add_argument("--checkpoint", default=None, help="断点文件，默认 <output>.ckpt.jsonl")
    data.add_argument("--start-idx", type=int, default=0, help="起始下标（含）")
    data.add_argument("--end-idx", type=int, default=None, help="结束下标（不含）")
    data.add_argument("--max-samples", type=int, default=None, help="最多处理条数")
    data.add_argument("--no-resume", action="store_true", help="忽略 checkpoint 重新跑")

    run = parser.add_argument_group("运行")
    run.add_argument("--mode", choices=["fast", "quality"], default="quality", help="合成模式")
    run.add_argument("--shard-size", type=int, default=500, help="每个 shard 的样本数")
    run.add_argument("--shard-interval", type=float, default=0.0, help="shard 间暂停秒数")
    run.add_argument("--workers", type=int, default=4, help="并发数")
    run.add_argument("--max-retries", type=int, default=3, help="单次调用重试次数")
    run.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    run.add_argument("--judge-temperature", type=float, default=0.1, help="裁判温度")
    run.add_argument("--dry-run", action="store_true", help="离线模式，不调用 API")

    models = parser.add_argument_group("模型")
    models.add_argument("--generator", type=parse_role, default=None, help="model@base_url[@key_env]")
    models.add_argument("--reviewer", type=parse_role, default=None, help="model@base_url[@key_env]")
    models.add_argument("--rewriter", type=parse_role, default=None, help="model@base_url[@key_env]")
    models.add_argument("--judge", type=parse_role, action="append", default=None,
                        help="可重复；多裁判默认全部通过才算通过")
    models.add_argument("--min-passing", type=int, default=None, help="至少多少个裁判通过")

    return parser


def main() -> int:
    args = build_parser().parse_args()

    seeds = load_seeds(args.seed_data)
    if not seeds:
        print(f"[data_pipeline] 未从 {args.seed_data} 读到任何种子数据", file=sys.stderr)
        return 1

    if args.end_idx is not None:
        seeds = seeds[args.start_idx:args.end_idx]
    else:
        seeds = seeds[args.start_idx:]
    if args.max_samples is not None:
        seeds = seeds[:args.max_samples]
    if not seeds:
        print("[data_pipeline] 切片后没有样本", file=sys.stderr)
        return 1

    config = PipelineConfig(
        mode=args.mode,
        shard_size=args.shard_size,
        shard_interval=args.shard_interval,
        max_workers=args.workers,
        temperature=args.temperature,
        judge_temperature=args.judge_temperature,
        max_retries=args.max_retries,
        generator=args.generator or DEFAULT_ROLES["generator"],
        reviewer=args.reviewer or DEFAULT_ROLES["reviewer"],
        rewriter=args.rewriter or DEFAULT_ROLES["rewriter"],
        judges=args.judge or [DEFAULT_ROLES["judge"]],
        min_passing_judges=args.min_passing,
        dry_run=args.dry_run,
    )

    output = Path(args.output)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output.with_suffix(output.suffix + ".ckpt.jsonl")

    departments = Counter(seed.department for seed in seeds)
    shards = make_shards(seeds, config.shard_size)

    print("=" * 60)
    print("MedAgent 数据合成")
    print("=" * 60)
    print(f"种子: {args.seed_data}（{len(seeds)} 条，{len(departments)} 个科室）")
    print(f"科室分布: {dict(departments.most_common(6))}")
    print(f"模式: {config.mode}  并发: {config.max_workers}  shard: {len(shards)}")
    print(f"生成: {config.generator.model}  复核: {config.reviewer.model}  "
          f"修订: {config.rewriter.model}")
    print(f"裁判: {[role.model for role in config.judges]}（需 {config.min_passing_judges} 票）")
    print(f"输出: {output}")
    print(f"断点: {checkpoint}  续跑: {not args.no_resume}")
    if config.dry_run:
        print("!! dry-run：不调用任何 API，仅验证管线与校验逻辑")
    print("=" * 60)

    stats = run_pipeline(
        seeds=seeds,
        config=config,
        output_path=output,
        checkpoint_path=checkpoint,
        resume=not args.no_resume,
    )

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats.get("passed", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())