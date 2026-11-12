"""离线演示用的脚本化医生策略

不调用任何 API，按"问诊 → 检查 → 查知识库 → 结论"的顺序动作，
``skill`` 控制最终结论的质量，用于 WebUI / CLI / 测试。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .env import MedAgentEnv


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


def scripted_action(env: MedAgentEnv, skill: float = 0.9) -> str:
    """根据环境当前状态给出下一步动作"""
    if env.state is None:
        raise RuntimeError("请先调用 reset()")
    state = env.state
    case = state.case

    disclosed_sub = state.disclosed_subjective
    disclosed_obj = state.disclosed_objective

    # 1) 先补主观信息，最多两轮
    if len(disclosed_sub) < 2 and state.action_counts["ASK"] < 2:
        keywords: List[str] = []
        for index, item in enumerate(case.subjective):
            if index in disclosed_sub:
                continue
            keywords.extend(item.keywords[:1])
            if len(keywords) >= 2:
                break
        return _fmt("ASK", keywords)

    # 2) 再开必要的检查，一次开两项
    if len(disclosed_obj) < 2 and state.action_counts["EXAM"] < 2:
        keywords = []
        for index, item in enumerate(case.objective):
            if index in disclosed_obj:
                continue
            if item.necessity:
                keywords.append(item.keywords[0])
            if len(keywords) >= 2:
                break
        return _fmt("EXAM", keywords or [case.objective[0].keywords[0]])

    # 3) 查一次知识库
    if state.action_counts["KNOWLEDGE"] == 0:
        keywords = getattr(case, "knowledge_keywords", None) or ["诊断标准", "治疗原则"]
        return _fmt("KNOWLEDGE", keywords[:2])

    # 4) 给结论
    gt = case.ground_truth
    diagnosis = gt.diagnosis[0] if gt.diagnosis else "待明确"
    if skill >= 0.75:
        treatment = "；".join(gt.treatment)
    elif skill >= 0.5:
        treatment = gt.treatment[0] if gt.treatment else "对症处理"
    else:
        diagnosis = "待进一步检查明确"
        treatment = "建议住院观察，完善检查"

    return _fmt("FINAL", diagnosis=diagnosis, treatment=treatment)


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