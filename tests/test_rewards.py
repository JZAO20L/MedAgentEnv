"""奖励组件与 swift ORM 封装测试"""

import pytest

from src.medagent.rewards import (
    cost_penalty,
    diagnosis_reward,
    efficiency_reward,
    safety_reward,
    terminal_breakdown,
    treatment_reward,
)
from src.medagent.swift_rewards import (
    MedCostReward,
    MedDiagnosisReward,
    MedEfficiencyReward,
    MedSafetyReward,
    MedTreatmentReward,
    register,
)


FINAL_COMPLETION = [{
    "role": "assistant",
    "content": (
        "<act><action>FINAL</action>"
        "<diagnosis>急性前壁心肌梗死</diagnosis>"
        "<treat>急诊PCI；双联抗血小板治疗</treat></act>"
    ),
}]

MULTI_TURN_COMPLETION = [
    {"role": "assistant", "content": "<act><action>ASK</action><keywords>疼痛性质, 持续时间</keywords></act>"},
    {"role": "user", "content": "压榨样疼痛。"},
    {"role": "assistant", "content": "<act><action>EXAM</action><keywords>心电图</keywords></act>"},
    {"role": "assistant", "content": "<act><action>FINAL</action><diagnosis>急性心肌梗死</diagnosis><treat>急诊PCI</treat></act>"},
]


def test_match_ratio_handles_empty_targets():
    assert diagnosis_reward("急性心肌梗死", []) == 0.0
    assert diagnosis_reward("急性心肌梗死", ["急性心肌梗死"]) == 1.0


def test_partial_treatment_match():
    score = treatment_reward("急诊PCI", ["急诊PCI", "抗凝治疗", "双联抗血小板治疗"])
    assert score == pytest.approx(1 / 3)


def test_safety_reward():
    assert safety_reward("急诊PCI", ["运动负荷试验"]) == 1.0
    assert safety_reward("建议运动负荷试验", ["运动负荷试验"]) == 0.0
    assert safety_reward("任何处置", []) == 1.0


def test_terminal_breakdown_weights():
    breakdown = terminal_breakdown(
        "急性前壁心肌梗死",
        "急诊PCI；双联抗血小板治疗；抗凝治疗",
        {
            "diagnosis": ["急性前壁心肌梗死"],
            "treatment": ["急诊PCI", "双联抗血小板治疗", "抗凝治疗"],
            "avoid": ["运动负荷试验"],
        },
    )
    assert breakdown["diagnosis"] == 1.0
    assert breakdown["treatment"] == 1.0
    assert breakdown["safety"] == 1.0
    assert breakdown["score"] == pytest.approx(1.0)


def test_efficiency_reward_bounds():
    assert efficiency_reward(0, 0, 0) == -0.5
    assert efficiency_reward(1, 1, 0) == 1.0
    assert efficiency_reward(2, 2, 2) < 1.0
    assert efficiency_reward(5, 5, 5) == -0.2


def test_cost_penalty_clipped():
    assert cost_penalty(0) == 0.0
    assert cost_penalty(125) == pytest.approx(-0.125)
    assert cost_penalty(10_000) == -1.0


def test_swift_diagnosis_reward():
    rewards = MedDiagnosisReward()(
        completions=[FINAL_COMPLETION],
        ground_truth_diagnosis=[["急性前壁心肌梗死"]],
    )
    assert rewards == [1.0]


def test_swift_treatment_and_safety_reward():
    treatment = MedTreatmentReward()(
        completions=[FINAL_COMPLETION],
        ground_truth_treatment=[["急诊PCI", "双联抗血小板治疗"]],
    )
    safety = MedSafetyReward()(
        completions=[FINAL_COMPLETION],
        ground_truth_avoid=[["运动负荷试验"]],
    )
    assert treatment == [1.0]
    assert safety == [1.0]


def test_swift_efficiency_counts_actions():
    rewards = MedEfficiencyReward()(completions=[MULTI_TURN_COMPLETION])
    assert rewards == [1.0]


def test_swift_cost_reward_is_non_positive():
    rewards = MedCostReward()(completions=[MULTI_TURN_COMPLETION])
    assert rewards[0] <= 0.0
    assert rewards[0] >= -1.0


def test_swift_reward_handles_string_completions():
    rewards = MedDiagnosisReward()(
        completions=["<act><action>FINAL</action><diagnosis>急性心肌梗死</diagnosis></act>"],
        ground_truth_diagnosis=[["急性心肌梗死"]],
    )
    assert rewards == [1.0]


def test_register_custom_registry():
    registry = {}
    register(registry)
    assert set(registry) == {"med_diagnosis", "med_treatment", "med_safety", "med_efficiency", "med_cost"}