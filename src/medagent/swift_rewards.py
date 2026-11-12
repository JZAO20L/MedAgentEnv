"""把医疗奖励封装成 ms-swift 的 ORM

用法（训练侧）：

    swift rlhf ... --external_plugins scripts/swift_plugin.py \
        --reward_funcs med_diagnosis med_treatment med_safety med_efficiency med_cost

每个函数返回 ``List[float]``，与 ``completions`` 对齐；数据集列通过 kwargs 传入。
本模块不导入 swfit 也能工作（提供最小 ORM 基类），方便单测。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

try:  # pragma: no cover - 训练环境才有 swift
    from swift.rewards import ORM, orms
    SWIFT_AVAILABLE = True
except Exception:  # noqa: BLE001
    class ORM:  # type: ignore[no-redef]
        """swift 未安装时的最小占位基类"""

    orms: Dict[str, Any] = {}
    SWIFT_AVAILABLE = False

from .env import last_assistant_text, parse_action
from .rewards import (
    ACTION_COST,
    as_list,
    cost_penalty,
    diagnosis_reward,
    efficiency_reward,
    safety_reward,
    treatment_reward,
)

_ACTION_PATTERN = re.compile(r"<action>\s*([A-Za-z]+)\s*</action>", re.IGNORECASE)
_KEYWORDS_PATTERN = re.compile(r"<keywords>(.*?)</keywords>", re.DOTALL | re.IGNORECASE)


def _column(kwargs: Dict[str, Any], name: str, index: int) -> Any:
    values = kwargs.get(name)
    if not isinstance(values, (list, tuple)):
        return values
    return values[index] if index < len(values) else None


def _full_text(completion: Any) -> str:
    """整条轨迹的文本（效率/成本需要统计所有动作）"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = [
            str(message.get("content", ""))
            for message in completion
            if isinstance(message, dict) and message.get("content")
        ]
        return "\n".join(parts)
    return str(completion)


def _predicted_treatment(text: str) -> str:
    parsed = parse_action(text)
    return parsed.get("treatment") or text


def _action_counts(text: str) -> Dict[str, int]:
    counts = {"ASK": 0, "EXAM": 0, "KNOWLEDGE": 0, "FINAL": 0}
    for match in _ACTION_PATTERN.finditer(text or ""):
        action = match.group(1).upper()
        if action in counts:
            counts[action] += 1
    return counts


def _estimated_cost(text: str) -> float:
    """按动作标签与 keywords 数估算成本"""
    total = 0.0
    for block in re.findall(r"<act>(.*?)</act>", text or "", re.DOTALL | re.IGNORECASE):
        match = _ACTION_PATTERN.search(block)
        if not match:
            continue
        action = match.group(1).upper()
        if action not in ACTION_COST:
            continue
        keywords_match = _KEYWORDS_PATTERN.search(block)
        keywords = [item for item in re.split(r"[,，、;；]+", keywords_match.group(1)) if item.strip()] if keywords_match else []
        total += ACTION_COST[action] * max(1, len(keywords))
    return total


class _MedRewardBase(ORM):
    """统一处理 completions → 文本，再交给子类打分"""

    use_full_trajectory = False

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            text = _full_text(completion) if self.use_full_trajectory else last_assistant_text(completion)
            rewards.append(float(self.score(text, index, kwargs)))
        return rewards

    def score(self, text: str, index: int, kwargs: Dict[str, Any]) -> float:
        raise NotImplementedError


class MedDiagnosisReward(_MedRewardBase):
    def score(self, text: str, index: int, kwargs: Dict[str, Any]) -> float:
        prediction = parse_action(text).get("diagnosis") or text
        return diagnosis_reward(prediction, as_list(_column(kwargs, "ground_truth_diagnosis", index)))


class MedTreatmentReward(_MedRewardBase):
    def score(self, text: str, index: int, kwargs: Dict[str, Any]) -> float:
        prediction = _predicted_treatment(text)
        return treatment_reward(prediction, as_list(_column(kwargs, "ground_truth_treatment", index)))


class MedSafetyReward(_MedRewardBase):
    def score(self, text: str, index: int, kwargs: Dict[str, Any]) -> float:
        prediction = _predicted_treatment(text)
        return safety_reward(prediction, as_list(_column(kwargs, "ground_truth_avoid", index)))


class MedEfficiencyReward(_MedRewardBase):
    use_full_trajectory = True

    def score(self, text: str, index: int, kwargs: Dict[str, Any]) -> float:
        counts = _action_counts(text)
        return efficiency_reward(counts["ASK"], counts["EXAM"], counts["KNOWLEDGE"])


class MedCostReward(_MedRewardBase):
    use_full_trajectory = True

    def score(self, text: str, index: int, kwargs: Dict[str, Any]) -> float:
        return cost_penalty(_estimated_cost(text))


DEFAULT_REGISTRY = {
    "med_diagnosis": MedDiagnosisReward,
    "med_treatment": MedTreatmentReward,
    "med_safety": MedSafetyReward,
    "med_efficiency": MedEfficiencyReward,
    "med_cost": MedCostReward,
}


def register(registry: Dict[str, Any] = None) -> Dict[str, Any]:
    target = registry if registry is not None else orms
    for name, cls in DEFAULT_REGISTRY.items():
        target[name] = cls
    return target


register()