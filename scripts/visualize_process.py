#!/usr/bin/env python3
"""离线过程可视化：回放一条完整 agent loop 及其多维反馈，生成自包含 HTML。

    python scripts/visualize_process.py
    python scripts/visualize_process.py --case-id mock-mi-001 --skill 0.9 --out docs/process_demo.html

页面内容（环境的核心机制，不做水平分档对比）：
- ① agent loop：主诉 → ASK / EXAM / KNOWLEDGE → FINAL 的逐轮轨迹，
  每轮标注动作、关键词、披露信息、奖励拆解（新颖度 − 成本）与累计成本曲线
- ② 多维反馈：逐轮 / 终止奖励公式 + 诊断 / 治疗 / 安全 / 效率四维拆解 + 成本与轮次
- ③ 结论对照：agent 诊断 / 处置 vs 标准答案与禁忌清单

不调用任何 API，数据来自 assets/mock/；同一输入生成的页面内容确定，可入库。
"""

from __future__ import annotations

import argparse
import asyncio
import html
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.medagent.demo_agent import scripted_action
from src.medagent.env import CaseCatalog, MedAgentEnv
from src.medagent.rewards import ACTION_COST, COST_SCALE

ACTION_COLOR = {"ASK": "#0f766e", "EXAM": "#b45309", "KNOWLEDGE": "#4a6fa5", "FINAL": "#182430"}
ACTION_LABEL = {"ASK": "问诊", "EXAM": "检查", "KNOWLEDGE": "查知识", "FINAL": "结论"}

CSS = """
:root{--ink:#182430;--muted:#68757f;--line:#dfe5ea;--panel:#f5f7f8;
--accent:#0f766e;--accent-soft:#e2f0ee;--amber:#b45309;--amber-soft:#fbf0e0;
--mono:"Consolas","SFMono-Regular",monospace}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);
font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",sans-serif;
font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased}
main{max-width:1080px;margin:0 auto;padding:32px 24px 60px}
header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:24px}
.tag{display:inline-block;font-size:12px;letter-spacing:.1em;color:var(--accent);
background:var(--accent-soft);border-radius:999px;padding:4px 12px;margin-bottom:12px}
h1{font-size:28px;line-height:1.3;margin:0 0 8px}
h2{font-size:19px;margin:30px 0 14px}
.muted{color:var(--muted)} .small{font-size:12.5px}
.case-meta{display:flex;gap:22px;flex-wrap:wrap;font-size:13.5px;color:var(--muted);margin-top:8px}
.case-meta b{color:var(--ink)}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .big{font-family:var(--mono);font-size:26px;font-weight:700;line-height:1.15}
.card .cap{font-size:12.5px;color:var(--muted);margin-top:4px}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:12px}
.tile{background:var(--panel);border-radius:10px;padding:10px 12px}
.tile .n{font-family:var(--mono);font-size:20px;font-weight:700}
.tile .t{font-size:11.5px;color:var(--muted)}
svg{width:100%;height:auto;display:block;margin:6px 0 14px}
.turn{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px;background:#fff}
.turn-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{font-size:11.5px;font-weight:700;color:#fff;border-radius:6px;padding:2px 8px;white-space:nowrap}
.reward{font-family:var(--mono);font-size:12px;font-weight:700}
.reward.pos{color:var(--accent)} .reward.neg{color:#b3261e}
.cum{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--amber);white-space:nowrap}
.kw{margin-top:6px}
.kw span{display:inline-block;font-size:11.5px;background:var(--panel);border:1px solid var(--line);
border-radius:999px;padding:1px 9px;margin:2px 4px 2px 0;color:#3d4b57}
.formula{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:5px}
.obs{font-size:12.5px;color:#3d4b57;margin-top:6px;white-space:pre-line}
.tmeta{font-size:11.5px;color:var(--muted);margin-top:5px}
.brow{display:flex;align-items:center;gap:8px;font-size:12.5px;margin-bottom:6px}
.brow>span{flex:0 0 34px;color:var(--muted)}
.brow b{font-family:var(--mono);font-size:12px;flex:0 0 38px;text-align:right}
.track{flex:1;height:8px;background:#e9edf1;border-radius:999px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:999px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#eef1f4;font-size:12px;color:#3d4b57;white-space:nowrap}
tr:last-child td{border-bottom:none}
.note{font-size:12.5px;color:var(--muted);border-top:1px dashed var(--line);padding-top:12px;margin-top:24px}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border-radius:4px;padding:1px 5px}
@media (max-width:860px){.g2{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
@media print{body{font-size:11pt}.col,.card,.turn{break-inside:avoid}main{padding:0}}
"""


def _clean_observation(text: str, limit: int = 160) -> str:
    body = text.split("\n\n（本轮成本")[0].strip()
    if len(body) > limit:
        body = body[:limit] + "…"
    return body


def _reward_note(action: str, turn: Dict[str, Any], prev_cost: float,
                 score: float, total_cost: float) -> str:
    """把该轮奖励写成可手工验算的拆解式"""
    delta = turn["cost"] - prev_cost
    if action in ("ASK", "EXAM"):
        novelty = min(0.3, 0.1 * (turn["new_items"] or 0))
        note = f"新颖度 {novelty:.2f} − 成本 {delta:.0f}×{COST_SCALE}"
        if not turn["new_items"]:
            note += " − 重复 0.05"
        return note
    if action == "KNOWLEDGE":
        base = 0.05 if turn["hits"] else -0.05
        return f"命中 {base:+.2f} − 成本 {delta:.0f}×{COST_SCALE}"
    return f"得分 {score:.2f} − 累计成本 {total_cost:.0f}×{COST_SCALE}（裁剪 [-1,1]）"


async def _trace(case_id: str, skill: float, max_steps: int) -> Dict[str, Any]:
    """驱动环境跑完一条轨迹，逐轮采集 info（动作 / 新信息 / 奖励 / 成本）"""
    env = MedAgentEnv(env_config={"case_id": case_id, "max_steps": max_steps})
    await env.reset()
    turns: List[Dict[str, Any]] = []
    prev_cost = 0.0
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
        prev_cost = info["cost"]
        if done:
            break
    state = env.state
    score = round(state.breakdown.get("score", 0.0), 4)
    for index, turn in enumerate(turns):
        before = turns[index - 1]["cost"] if index else 0.0
        turn["reward_note"] = _reward_note(turn["action"], turn, before, score, state.cost)
    return {
        "skill": skill,
        "turns": turns,
        "score": score,
        "cost": round(state.cost, 2),
        "total_reward": round(state.total_reward, 4),
        "breakdown": {key: round(value, 4) for key, value in state.breakdown.items()},
        "diagnosis": state.diagnosis,
        "treatment": state.treatment,
        "done_reason": state.done_reason,
        "ground_truth": {
            "diagnosis": state.case.ground_truth.diagnosis,
            "treatment": state.case.ground_truth.treatment,
            "avoid": state.case.ground_truth.avoid,
        },
    }


def build_data(case_id: str = "mock-mi-001", skill: float = 0.9,
               max_steps: int = 16) -> Dict[str, Any]:
    catalog = CaseCatalog()
    if case_id not in {item["case_id"] for item in catalog.list()}:
        raise SystemExit(f"病例不存在: {case_id}（可用: python scripts/cli.py cases）")
    trace = asyncio.run(_trace(case_id, skill, max_steps))
    case = catalog.get(case_id)
    return {
        "case": {
            "case_id": case.case_id,
            "chief_complaint": case.chief_complaint,
            "department": case.tags[0] if case.tags else "未知",
            "difficulty": case.difficulty,
        },
        "trace": trace,
        "command": f"python scripts/visualize_process.py --case-id {case_id} --skill {skill}",
    }


def _cost_svg(trace: Dict[str, Any], color: str = "#b45309") -> str:
    turns = trace["turns"]
    width, height = 640, 110
    count = len(turns)
    ymax = max(50.0, math.ceil(max(turn["cost"] for turn in turns) / 50) * 50)
    xs = [40 + index * (width - 110) / max(1, count - 1) for index in range(count)]
    ys = [height - 24 - (turn["cost"] / ymax) * (height - 44) for turn in turns]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>'
        f'<text x="{x:.1f}" y="{y - 8:.1f}" font-size="10" fill="{color}" '
        f'text-anchor="middle" font-family="Consolas,monospace">{turn["cost"]:.0f}</text>'
        for x, y, turn in zip(xs, ys, turns))
    labels = "".join(
        f'<text x="{x:.1f}" y="{height - 8}" font-size="10" fill="#68757f" '
        f'text-anchor="middle" font-family="Consolas,monospace">T{turn["turn"]}</text>'
        for x, turn in zip(xs, turns))
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="累计成本曲线">'
        f'<line x1="40" y1="{height - 24}" x2="{width - 30}" y2="{height - 24}" stroke="#dfe5ea"/>'
        f'<text x="40" y="14" font-size="10" fill="#68757f" font-family="Consolas,monospace">'
        f'累计成本（元，满刻度 {ymax:.0f}）</text>'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        f'{dots}{labels}</svg>'
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
        f'<div class="formula">奖励 = {html.escape(turn["reward_note"])}</div>'
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


def render_html(data: Dict[str, Any]) -> str:
    case = data["case"]
    trace = data["trace"]
    turns_html = "".join(_turn_html(turn) for turn in trace["turns"])
    gt = trace["ground_truth"]
    gt_rows = (
        f"<tr><td>诊断</td><td>{html.escape(trace['diagnosis'] or '未给出')}</td>"
        f"<td>{html.escape('；'.join(gt['diagnosis']))}</td></tr>"
        f"<tr><td>处置</td><td>{html.escape(trace['treatment'] or '未给出')}</td>"
        f"<td>{html.escape('；'.join(gt['treatment']))}</td></tr>"
        f"<tr><td>禁忌</td><td colspan='2'>处置不得包含：{html.escape('；'.join(gt['avoid']) or '无')}</td></tr>"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedAgentEnv · agent loop 与多维反馈（{html.escape(case['case_id'])}）</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header>
  <span class="tag">MedAgentEnv · 离线过程可视化 · mock 数据</span>
  <h1>一条完整 agent loop，和它拿到的多维反馈</h1>
  <div class="case-meta">
    <span>病例 <b>{html.escape(case['case_id'])}</b></span>
    <span>科室 <b>{html.escape(case['department'])}</b></span>
    <span>难度 <b>{html.escape(str(case['difficulty']))}</b></span>
    <span>skill <b>{trace['skill']:.2f}</b></span>
    <span>主诉 <b>{html.escape(case['chief_complaint'])}</b></span>
  </div>
</header>

<h2>① Agent loop：主诉 → 获取信息 → 结论</h2>
{_cost_svg(trace)}
{turns_html}

<h2>② 多维反馈：每一轮和终止时各拿到什么</h2>
<div class="grid g2">
  <div class="card">
    <h3 style="margin:0 0 8px;font-size:15px">奖励怎么算（纯函数，零 API）</h3>
    <p class="small" style="margin:0 0 6px"><b>逐轮</b>：信息新颖度（每条新信息 +0.1，上限 0.3）− 动作成本×{COST_SCALE}；重复问同一件事不给新信息再 −0.05。</p>
    <p class="small" style="margin:0 0 6px"><b>终止</b>：综合得分（0.5×诊断 + 0.3×治疗 + 0.2×安全）− 累计成本×{COST_SCALE}。</p>
    <p class="small muted" style="margin:0">单轮与终止奖励均裁剪到 [-1, 1]，避免早期探索被成本压死；效率分要求至少各一次问诊与检查。</p>
  </div>
  <div class="card">
    <h3 style="margin:0 0 8px;font-size:15px">终止四维拆解（0–1）</h3>
    {_bars_html(trace['breakdown'])}
    <div class="tiles">
      <div class="tile"><div class="n" style="color:var(--accent)">{trace['score']:.2f}</div><div class="t">综合得分</div></div>
      <div class="tile"><div class="n" style="color:var(--amber)">{trace['cost']:.0f}</div><div class="t">累计成本（元）</div></div>
      <div class="tile"><div class="n">{trace['total_reward']:.2f}</div><div class="t">累计奖励</div></div>
      <div class="tile"><div class="n">{len(trace['turns'])}</div><div class="t">轮次</div></div>
    </div>
  </div>
</div>

<h2>③ 结论对照：打分依据</h2>
<div class="card" style="padding:0;overflow:hidden">
  <table>
    <tr><th style="width:10%">维度</th><th style="width:45%">agent 输出</th><th>标准答案</th></tr>
    {gt_rows}
  </table>
</div>

<div class="note">
  <p>数据与口径：病例来自 <code>assets/mock/</code>，医生为离线剧本策略（不调用 API）；
  成本为代理定价（问诊 {ACTION_COST['ASK']} 元 / 检查 {ACTION_COST['EXAM']} 元 / 查知识 {ACTION_COST['KNOWLEDGE']} 元），只用于相对比较。
  本页得分是环境纯函数规则分（0–1），与论文基准的 LLM 判分（1–5）不同量纲，不可混比。</p>
  <p>轨迹与反馈计算由 <code>tests/test_visualize_process.py</code> 锁定。
  重新生成：<code>{html.escape(data['command'])}</code></p>
</div>
</main>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedAgentEnv 离线过程可视化（agent loop + 多维反馈）")
    parser.add_argument("--case-id", default="mock-mi-001", help="回放的病例")
    parser.add_argument("--skill", type=float, default=0.9, help="剧本医生水平参数")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "process_demo.html"))
    args = parser.parse_args(argv)

    data = build_data(args.case_id, args.skill, args.max_steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")

    trace = data["trace"]
    print(f"saved {out}")
    print(f"  {args.case_id} skill={args.skill:.2f}: {len(trace['turns'])} 轮 | "
          f"得分 {trace['score']:.2f} | 成本 {trace['cost']:.0f} 元 | 累计奖励 {trace['total_reward']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())