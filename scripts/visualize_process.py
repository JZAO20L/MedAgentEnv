#!/usr/bin/env python3
"""离线过程可视化：用剧本医生跑 mock 病例，生成自包含 HTML（零外部依赖）。

    python scripts/visualize_process.py
    python scripts/visualize_process.py --case-id mock-mi-001 --skills 1.0,0.6,0.2 --out docs/process_demo.html

页面内容：
- 三档水平在全部 mock 病例上的平均得分 / 成本汇总条（与 ``cli.py sweep`` 同源）
- 选定病例三档的逐轮动作、关键词、信息披露、逐轮奖励与累计成本曲线
- 终止得分拆解（诊断 / 治疗 / 安全 / 效率）

不调用任何 API，数据来自 assets/mock/；同一输入生成的页面内容确定，可入库。
"""

from __future__ import annotations

import argparse
import asyncio
import html
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.medagent.demo_agent import run_episode, scripted_action, skill_tier
from src.medagent.env import CaseCatalog, MedAgentEnv

ACTION_COLOR = {"ASK": "#0f766e", "EXAM": "#b45309", "KNOWLEDGE": "#4a6fa5", "FINAL": "#182430"}
ACTION_LABEL = {"ASK": "问诊", "EXAM": "检查", "KNOWLEDGE": "查知识", "FINAL": "结论"}
TIER_COLOR = {"专家": "#0f766e", "主治": "#4aa39a", "实习生": "#b45309"}

CSS = """
:root{--ink:#182430;--muted:#68757f;--line:#dfe5ea;--panel:#f5f7f8;
--accent:#0f766e;--accent-soft:#e2f0ee;--amber:#b45309;--amber-soft:#fbf0e0;
--mono:"Consolas","SFMono-Regular",monospace}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);
font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",sans-serif;
font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased}
main{max-width:1180px;margin:0 auto;padding:32px 24px 60px}
header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:24px}
.tag{display:inline-block;font-size:12px;letter-spacing:.1em;color:var(--accent);
background:var(--accent-soft);border-radius:999px;padding:4px 12px;margin-bottom:12px}
h1{font-size:30px;line-height:1.3;margin:0 0 8px}
h2{font-size:20px;margin:32px 0 14px}
.muted{color:var(--muted)} .small{font-size:12.5px}
.case-meta{display:flex;gap:22px;flex-wrap:wrap;font-size:13.5px;color:var(--muted);margin-top:8px}
.case-meta b{color:var(--ink)}
.grid{display:grid;gap:14px}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .big{font-family:var(--mono);font-size:28px;font-weight:700;line-height:1.15}
.card .cap{font-size:12.5px;color:var(--muted);margin-top:4px}
.col{border:1px solid var(--line);border-radius:12px;padding:16px;background:#fff}
.col-head{display:flex;align-items:baseline;gap:10px;border-bottom:1px solid var(--line);
padding-bottom:10px;margin-bottom:12px}
.col-head .tier{font-size:17px;font-weight:700}
.col-head .skill{font-family:var(--mono);font-size:12px;color:var(--muted)}
.col-head .score{margin-left:auto;font-family:var(--mono);font-size:24px;font-weight:700}
.col-sub{font-size:12.5px;color:var(--muted);margin-bottom:10px}
svg{width:100%;height:auto;display:block;margin-bottom:12px}
.turn{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px;background:#fff}
.turn-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{font-size:11.5px;font-weight:700;color:#fff;border-radius:6px;padding:2px 8px;white-space:nowrap}
.reward{font-family:var(--mono);font-size:12px;font-weight:700}
.reward.pos{color:var(--accent)} .reward.neg{color:#b3261e}
.cum{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--amber);white-space:nowrap}
.kw{margin-top:6px}
.kw span{display:inline-block;font-size:11.5px;background:var(--panel);border:1px solid var(--line);
border-radius:999px;padding:1px 9px;margin:2px 4px 2px 0;color:#3d4b57}
.obs{font-size:12.5px;color:#3d4b57;margin-top:6px;white-space:pre-line}
.tmeta{font-size:11.5px;color:var(--muted);margin-top:5px}
.brow{display:flex;align-items:center;gap:8px;font-size:12.5px;margin-bottom:6px}
.brow>span{flex:0 0 34px;color:var(--muted)}
.brow b{font-family:var(--mono);font-size:12px;flex:0 0 38px;text-align:right}
.track{flex:1;height:8px;background:#e9edf1;border-radius:999px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:999px}
.note{font-size:12.5px;color:var(--muted);border-top:1px dashed var(--line);padding-top:12px;margin-top:24px}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border-radius:4px;padding:1px 5px}
@media (max-width:900px){.g3{grid-template-columns:1fr}}
@media print{body{font-size:11pt}.col,.card{break-inside:avoid}main{padding:0}}
"""


def _clean_observation(text: str, limit: int = 160) -> str:
    body = text.split("\n\n（本轮成本")[0].strip()
    if len(body) > limit:
        body = body[:limit] + "…"
    return body


async def _trace(case_id: str, skill: float, max_steps: int) -> Dict[str, Any]:
    """驱动环境跑完一条轨迹，逐轮采集 info（动作 / 新信息 / 奖励 / 成本）"""
    env = MedAgentEnv(env_config={"case_id": case_id, "max_steps": max_steps})
    await env.reset()
    turns: List[Dict[str, Any]] = []
    while not env.state.done and len(turns) < max_steps:
        action = scripted_action(env, skill=skill)
        observation, _, done, info = await env.step(action)
        turns.append({
            "turn": info["turn"],
            "action": info["action"],
            "keywords": info.get("keywords") or [],
            "new_items": info.get("new_items"),
            "hits": info.get("hits"),
            "reward": info["step_reward"],
            "cost": info["cost"],
            "observation": _clean_observation(observation),
        })
        if done:
            break
    state = env.state
    return {
        "skill": skill,
        "tier": skill_tier(skill),
        "turns": turns,
        "score": round(state.breakdown.get("score", 0.0), 4),
        "cost": round(state.cost, 2),
        "breakdown": {key: round(value, 4) for key, value in state.breakdown.items()},
        "diagnosis": state.diagnosis,
        "treatment": state.treatment,
        "done_reason": state.done_reason,
    }


def build_data(case_id: str = "mock-mi-001", skills: Sequence[float] = (1.0, 0.6, 0.2),
               n: int = 6, max_steps: int = 16) -> Dict[str, Any]:
    catalog = CaseCatalog()
    cases = catalog.list()[:n]
    if case_id not in {item["case_id"] for item in catalog.list()}:
        raise SystemExit(f"病例不存在: {case_id}（可用: python scripts/cli.py cases）")

    summary = []
    for skill in skills:
        results = [
            asyncio.run(run_episode(
                MedAgentEnv(env_config={"case_id": item["case_id"], "max_steps": max_steps}),
                skill=skill, max_turns=max_steps))
            for item in cases
        ]
        count = len(results)
        summary.append({
            "skill": skill,
            "tier": skill_tier(skill),
            "avg_score": round(sum(r["score"] for r in results) / count, 3),
            "avg_cost": round(sum(r["cost"] for r in results) / count),
            "avg_turns": round(sum(r["turns"] for r in results) / count, 1),
        })

    traces = [asyncio.run(_trace(case_id, skill, max_steps)) for skill in skills]
    case = catalog.get(case_id)
    return {
        "case": {
            "case_id": case.case_id,
            "chief_complaint": case.chief_complaint,
            "department": case.tags[0] if case.tags else "未知",
            "difficulty": case.difficulty,
        },
        "n_cases": len(cases),
        "summary": summary,
        "traces": traces,
        "command": (f"python scripts/visualize_process.py --case-id {case_id} "
                    f"--skills {','.join(str(s) for s in skills)}"),
    }


def _cost_svg(trace: Dict[str, Any], ymax: float, color: str) -> str:
    turns = trace["turns"]
    width, height = 320, 96
    count = len(turns)
    xs = [24 + index * (width - 64) / max(1, count - 1) for index in range(count)]
    ys = [height - 20 - (turn["cost"] / ymax) * (height - 36) for turn in turns]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>' for x, y in zip(xs, ys))
    label_x = min(xs[-1] + 6, width - 40) if xs else 24
    label_y = (ys[-1] + 4) if ys else height - 20
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="累计成本曲线">'
        f'<line x1="24" y1="{height - 20}" x2="{width - 16}" y2="{height - 20}" stroke="#dfe5ea"/>'
        f'<text x="24" y="14" font-size="10" fill="#68757f" font-family="Consolas,monospace">'
        f'累计成本（元，满刻度 {ymax:.0f}）</text>'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        f'{dots}'
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="11" font-weight="bold" '
        f'fill="{color}" font-family="Consolas,monospace">{trace["cost"]:.0f}</text>'
        f'</svg>'
    )


def _turn_html(turn: Dict[str, Any]) -> str:
    action = turn["action"]
    color = ACTION_COLOR.get(action, "#182430")
    label = ACTION_LABEL.get(action, action)
    reward = turn["reward"]
    reward_cls = "pos" if reward >= 0 else "neg"
    keywords = "".join(f"<span>{html.escape(str(k))}</span>" for k in turn["keywords"])
    keyword_block = f'<div class="kw">{keywords}</div>' if keywords else ""
    if action in ("ASK", "EXAM"):
        meta = f"新信息 {turn['new_items']} 条"
    elif action == "KNOWLEDGE":
        meta = f"知识库命中 {turn['hits']} 条"
    else:
        meta = "episode 结束"
    return (
        '<div class="turn">'
        f'<div class="turn-head"><span class="badge" style="background:{color}">'
        f'T{turn["turn"]} · {label} {html.escape(action)}</span>'
        f'<span class="reward {reward_cls}">{reward:+.3f}</span>'
        f'<span class="cum">累计 {turn["cost"]:.0f} 元</span></div>'
        f'{keyword_block}'
        f'<div class="obs">{html.escape(turn["observation"])}</div>'
        f'<div class="tmeta">{meta}</div>'
        '</div>'
    )


def _bars_html(breakdown: Dict[str, float]) -> str:
    rows = []
    for key, label in (("diagnosis", "诊断"), ("treatment", "治疗"),
                       ("safety", "安全"), ("efficiency", "效率")):
        value = breakdown.get(key, 0.0)
        width = max(0.0, min(1.0, value)) * 100
        rows.append(
            f'<div class="brow"><span>{label}</span>'
            f'<div class="track"><div class="fill" style="width:{width:.0f}%"></div></div>'
            f'<b>{value:.2f}</b></div>'
        )
    return "".join(rows)


def _column_html(trace: Dict[str, Any], ymax: float) -> str:
    tier = trace["tier"]
    color = TIER_COLOR.get(tier, "#0f766e")
    turns_html = "".join(_turn_html(turn) for turn in trace["turns"])
    return (
        '<div class="col">'
        f'<div class="col-head"><span class="tier" style="color:{color}">{html.escape(tier)}</span>'
        f'<span class="skill">skill={trace["skill"]:.2f}</span>'
        f'<span class="score" style="color:{color}">{trace["score"]:.3f}</span></div>'
        f'<div class="col-sub">{len(trace["turns"])} 轮 · 成本 {trace["cost"]:.0f} 元 · '
        f'终止原因 {html.escape(trace["done_reason"])}</div>'
        f'{_cost_svg(trace, ymax, color)}'
        f'{turns_html}'
        f'<div class="col-sub" style="margin-top:12px">终止得分拆解</div>'
        f'{_bars_html(trace["breakdown"])}'
        '</div>'
    )


def render_html(data: Dict[str, Any]) -> str:
    case = data["case"]
    traces = data["traces"]
    ymax = max(50.0, math.ceil(max(trace["cost"] for trace in traces) / 50) * 50)

    summary_cards = "".join(
        '<div class="card">'
        f'<div class="big" style="color:{TIER_COLOR.get(row["tier"], "#0f766e")}">{row["avg_score"]:.3f}</div>'
        f'<div class="cap"><b>{html.escape(row["tier"])}</b>（skill={row["skill"]:.2f}）平均总分<br>'
        f'平均成本 {row["avg_cost"]} 元 · 平均 {row["avg_turns"]} 轮</div>'
        '</div>'
        for row in data["summary"]
    )
    columns = "".join(_column_html(trace, ymax) for trace in traces)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedAgentEnv · 问诊过程可视化（{html.escape(case['case_id'])}）</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header>
  <span class="tag">MedAgentEnv · 离线过程可视化 · mock 数据</span>
  <h1>一条问诊轨迹是怎么被打分和计价的</h1>
  <div class="case-meta">
    <span>病例 <b>{html.escape(case['case_id'])}</b></span>
    <span>科室 <b>{html.escape(case['department'])}</b></span>
    <span>难度 <b>{html.escape(str(case['difficulty']))}</b></span>
    <span>主诉 <b>{html.escape(case['chief_complaint'])}</b></span>
  </div>
</header>

<h2>三档水平 · 全部 {data['n_cases']} 个 mock 病例平均（与 <code>cli.py sweep</code> 同源）</h2>
<div class="grid g3">{summary_cards}</div>

<h2>选定病例的逐轮过程（同一病例，三种水平）</h2>
<div class="grid g3">{columns}</div>

<div class="note">
  <p>数据与口径：病例来自 <code>assets/mock/</code>，医生为离线剧本策略（不调用 API）；
  成本为代理定价（问诊 10 元 / 检查 50 元 / 查知识 5 元，1 元 = 0.001 奖励），只用于相对比较；
  逐轮奖励 = 信息新颖度 − 动作成本，终止奖励 = 得分 − 成本折算，均裁剪到 [-1, 1]。</p>
  <p>汇总条数字由 <code>tests/test_demo_gradient.py</code> 与 <code>tests/test_visualize_process.py</code> 锁定。
  重新生成：<code>{html.escape(data['command'])}</code></p>
</div>
</main>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedAgentEnv 离线过程可视化")
    parser.add_argument("--case-id", default="mock-mi-001", help="逐轮过程展示的病例")
    parser.add_argument("--skills", default="1.0,0.6,0.2", help="逗号分隔的水平取值")
    parser.add_argument("--n", type=int, default=6, help="汇总条使用的病例数")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "process_demo.html"))
    args = parser.parse_args(argv)

    skills = tuple(float(item) for item in str(args.skills).split(",") if item.strip())
    data = build_data(args.case_id, skills, args.n, args.max_steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")

    print(f"saved {out}")
    for row in data["summary"]:
        print(f"  {row['tier']} (skill={row['skill']:.2f}): "
              f"平均得分 {row['avg_score']:.3f} | 平均成本 {row['avg_cost']} 元 | n={data['n_cases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())