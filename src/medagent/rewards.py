"""可复用的奖励组件

同时被三处使用：
- ``env.py``：每个 step 的即时奖励与终止奖励
- ``demo_agent.py`` / WebUI：展示分数拆解
- ``swift_rewards.py``：封装成 ms-swift 的 ORM

所有函数都是纯函数，不依赖 API。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

# 终止奖励权重：诊断 0.5 / 治疗 0.3 / 安全 0.2
TERMINAL_WEIGHTS: Dict[str, float] = {
    "diagnosis": 0.5,
    "treatment": 0.3,
    "safety": 0.2,
}

# 每个动作的计价（元），与 data/README 的估算口径一致
ACTION_COST: Dict[str, int] = {
    "ASK": 10,
    "EXAM": 50,
    "KNOWLEDGE": 5,
}

# 1 元折合多少奖励惩罚
COST_SCALE = 0.001

# 单轮奖励的上下限，避免早期探索被成本淹没
STEP_REWARD_RANGE = (-1.0, 1.0)


def as_list(value: Any) -> List[str]:
    """把列值规整成字符串列表（兼容 swift 传 list / str / None）"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def match_ratio(prediction: str, targets: Sequence[str]) -> float:
    """targets 中被 prediction 覆盖的比例；targets 为空时返回 0"""
    targets = [target for target in targets if target]
    if not targets:
        return 0.0

    text = (prediction or "").lower()
    hits = 0
    for target in targets:
        needle = target.lower().strip()
        if len(needle) >= 2 and needle in text:
            hits += 1
    return hits / len(targets)


def diagnosis_reward(prediction: str, targets: Sequence[str]) -> float:
    return match_ratio(prediction, targets)


def treatment_reward(prediction: str, targets: Sequence[str]) -> float:
    return match_ratio(prediction, targets)


def safety_reward(treatment_text: str, avoid: Sequence[str]) -> float:
    """未触碰禁忌得 1，触碰任意一条得 0"""
    avoid = [item for item in avoid if item]
    if not avoid:
        return 1.0
    text = (treatment_text or "").lower()
    for item in avoid:
        needle = item.lower().strip()
        if len(needle) >= 2 and needle in text:
            return 0.0
    return 1.0


def terminal_breakdown(
    diagnosis: str,
    treatment: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """终止时的四路拆解：诊断 / 治疗 / 安全 / 加权总分"""
    gt_diagnosis = as_list(ground_truth.get("diagnosis"))
    gt_treatment = as_list(ground_truth.get("treatment"))
    gt_avoid = as_list(ground_truth.get("avoid"))

    parts = {
        "diagnosis": diagnosis_reward(diagnosis, gt_diagnosis),
        "treatment": treatment_reward(treatment, gt_treatment),
        "safety": safety_reward(treatment, gt_avoid),
    }
    parts["score"] = sum(TERMINAL_WEIGHTS[name] * parts[name] for name in TERMINAL_WEIGHTS)
    return parts


def efficiency_reward(n_ask: int, n_exam: int, n_knowledge: int) -> float:
    """工具使用效率：至少各一次 ASK/EXAM，调用越少越高，越界给负分"""
    total = n_ask + n_exam + n_knowledge
    if n_ask == 0 or n_exam == 0:
        return -0.5
    if total <= 2:
        return 1.0
    if total <= 6:
        return 0.8 - 0.1 * (total - 3)
    return -0.2


def cost_penalty(cost_yuan: float) -> float:
    """诊疗成本惩罚，落在 [-1, 0]"""
    penalty = -float(cost_yuan) * COST_SCALE
    return max(-1.0, min(0.0, penalty))


def format_reward(text: str) -> float:
    """动作格式是否合法（给 0 / 0.1 的稀疏信号）"""
    if not text:
        return 0.0
    lowered = text.lower()
    if "<act>" in lowered and "</act>" in lowered and "<action>" in lowered:
        return 0.1
    return 0.0


def clip_step_reward(value: float) -> float:
    low, high = STEP_REWARD_RANGE
    return max(low, min(high, value))