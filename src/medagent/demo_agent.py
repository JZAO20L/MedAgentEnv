"""离线演示用的脚本化医生策略

不调用任何 API，按"问诊 → 检查 → 查知识库 → 结论"的顺序动作。
``skill`` 分三档，差异体现在三处，因此同一病例的得分与成本随水平单调变化：

| 档位 | 诊断 | 处置 | 检查 |
|------|------|------|------|
| 专家 (>=0.8) | 直接给出标准诊断 | 覆盖全部处置要点 | 只开必要检查 |
| 主治 (>=0.5) | 加"考虑…可能"的限定语 | 只覆盖一半处置要点 | 必要检查 + 一项非必要检查 |
| 实习生 (<0.5) | 结论含糊，不给明确诊断 | 仅对症处理 | 全套检查 |

因为策略是确定的，同一 ``(case_id, skill)`` 的轨迹和得分可复现，用于 WebUI 演示与回归测试。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .env import MedAgentEnv

# 档位：(名称, 下界)，按 skill 从高到低匹配
SKILL_TIERS = (
    ("专家", 0.8),
    ("主治", 0.5),
    ("实习生", 0.0),
)


def skill_tier(skill: float) -> str:
    """把 0~1 的连续水平映射到三档医生"""
    for name, lower in SKILL_TIERS:
        if skill >= lower:
            return name
    return SKILL_TIERS[-1][0]


def _fmt(action: str, keywords: Optional[List[str]] = None,
         diagnosis: str = "", treatment: str = "") -> str:
    if action == "FINAL":
        return (
            "<act>\n<action>FINAL</action>\n"
            f"<diagnosis>{diagnosis}</diagnosis>\n"
            f"<treat>{treatment}</treat>\n</act>"
        )
    joined = ", ".join(keywords or [])
    return f"<act>\n<action>{action}</action>\n<keywords>{joined}</keywords>\n</act>"


def _exam_keywords(case: Any, tier: str) -> List[str]:
    """按档位决定这次检查开哪几项"""
    necessary = [item for item in case.objective if item.necessity]
    optional = [item for item in case.objective if not item.necessity]

    if tier == "专家":
        chosen = necessary[:2]
    elif tier == "主治":
        chosen = necessary[:2] + optional[:1]
    else:
        chosen = list(case.objective)

    keywords = [item.keywords[0] for item in chosen if item.keywords]
    return keywords or ([case.objective[0].keywords[0]] if case.objective and case.objective[0].keywords else [])


def _final_text(case: Any, tier: str) -> str:
    """按档位给出结论：这是得分梯度的主要来源"""
    gt = case.ground_truth
    diagnosis = gt.diagnosis[0] if gt.diagnosis else "待明确"
    treatments = list(gt.treatment)

    if tier == "专家":
        return _fmt("FINAL", diagnosis=diagnosis, treatment="；".join(treatments))

    if tier == "主治":
        keep = max(1, math.ceil(len(treatments) / 2))
        return _fmt(
            "FINAL",
            diagnosis=f"考虑{diagnosis}可能",
            treatment="；".join(treatments[:keep]),
        )

    return _fmt(
        "FINAL",
        diagnosis="待进一步检查明确",
        treatment="对症支持治疗，完善检查后决定",
    )


def scripted_action(env: MedAgentEnv, skill: float = 0.9) -> str:
    """根据环境当前状态给出下一步动作"""
    if env.state is None:
        raise RuntimeError("请先调用 reset()")
    state = env.state
    case = state.case
    tier = skill_tier(skill)

    # 1) 先补主观信息，最多两轮（各档一致：问诊是基本功）
    if len(state.disclosed_subjective) < 2 and state.action_counts["ASK"] < 2:
        keywords: List[str] = []
        for index, item in enumerate(case.subjective):
            if index in state.disclosed_subjective:
                continue
            keywords.extend(item.keywords[:1])
            if len(keywords) >= 2:
                break
        return _fmt("ASK", keywords)

    # 2) 再开检查：档位差异在这里体现为成本
    if not state.disclosed_objective and state.action_counts["EXAM"] == 0:
        return _fmt("EXAM", _exam_keywords(case, tier))

    # 3) 查一次知识库
    if state.action_counts["KNOWLEDGE"] == 0:
        keywords = getattr(case, "knowledge_keywords", None) or ["诊断标准", "治疗原则"]
        return _fmt("KNOWLEDGE", keywords[:2])

    # 4) 给结论：档位差异在这里体现为得分
    return _final_text(case, tier)


async def run_episode(env: MedAgentEnv, skill: float = 0.9, max_turns: int = 12) -> Dict[str, Any]:
    """跑完一个 episode，返回可直接给 CLI / WebUI 用的结构"""
    await env.reset()
    total_reward = 0.0
    turns = 0

    while not env.state.done and turns < max_turns:
        action = scripted_action(env, skill=skill)
        _, reward, done, _ = await env.step(action)
        total_reward += reward
        turns += 1
        if done:
            break

    state = env.state
    return {
        "case": {
            "case_id": state.case.case_id,
            "chief_complaint": state.case.chief_complaint,
            "department": state.case.tags[0] if state.case.tags else "未知",
            "difficulty": state.case.difficulty,
        },
        "skill": skill,
        "tier": skill_tier(skill),
        "transcript": state.transcript,
        "turns": turns,
        "total_reward": round(total_reward, 4),
        "cost": round(state.cost, 2),
        "score": round(state.breakdown.get("score", 0.0), 4),
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