"""医生水平梯度回归测试

答辩演示依赖一个卖点：拉低"医生水平"，得分下降、成本上升。
这里把这条性质固化成断言，避免以后改奖励或改剧本时悄悄失效。
"""

import asyncio

import pytest

from scripts.cli import main as cli_main
from src.medagent.demo_agent import run_episode, skill_tier
from src.medagent.env import CaseCatalog, MedAgentEnv

CASE_IDS = [item["case_id"] for item in CaseCatalog().list()]


def _run(case_id: str, skill: float) -> dict:
    env = MedAgentEnv(env_config={"case_id": case_id})
    return asyncio.run(run_episode(env, skill=skill))


def _avg(skill: float, key: str) -> float:
    values = [_run(case_id, skill)[key] for case_id in CASE_IDS]
    return sum(values) / len(values)


def test_skill_tier_boundaries():
    assert skill_tier(1.0) == "专家"
    assert skill_tier(0.8) == "专家"
    assert skill_tier(0.79) == "主治"
    assert skill_tier(0.5) == "主治"
    assert skill_tier(0.49) == "实习生"
    assert skill_tier(0.0) == "实习生"


def test_score_drops_monotonically_with_skill():
    expert = _avg(0.9, "score")
    resident = _avg(0.6, "score")
    intern = _avg(0.2, "score")

    assert expert > resident > intern
    assert expert == pytest.approx(0.958, abs=0.01)
    assert intern == pytest.approx(0.2, abs=0.01)


def test_cost_rises_as_skill_drops():
    expert = _avg(0.9, "cost")
    resident = _avg(0.6, "cost")
    intern = _avg(0.2, "cost")

    assert expert < resident < intern
    assert expert == pytest.approx(125, abs=1)
    assert intern == pytest.approx(175, abs=1)


def test_breakdown_differs_by_tier_on_one_case():
    expert = _run("mock-mi-001", 0.9)["breakdown"]
    resident = _run("mock-mi-001", 0.6)["breakdown"]
    intern = _run("mock-mi-001", 0.2)["breakdown"]

    assert expert["diagnosis"] == 1.0
    assert expert["treatment"] == 1.0
    assert resident["diagnosis"] == 1.0
    assert resident["treatment"] == pytest.approx(2 / 3, abs=0.01)
    assert intern["diagnosis"] == 0.0
    assert intern["treatment"] == 0.0
    assert intern["safety"] == 1.0


def test_resident_exam_is_broader_than_expert():
    expert = _run("mock-mi-001", 0.9)
    resident = _run("mock-mi-001", 0.6)

    assert resident["cost"] > expert["cost"]
    assert resident["turns"] == expert["turns"]


def test_webui_run_reflects_skill():
    from src.medagent import webui

    good = webui.handle_run({"case_id": "mock-mi-001", "skill": 0.9})
    bad = webui.handle_run({"case_id": "mock-mi-001", "skill": 0.2})

    assert good["score"] > bad["score"]
    assert good["tier"] == "专家"
    assert bad["tier"] == "实习生"


def test_sweep_command_prints_tier_summary(capsys):
    code = cli_main(["sweep", "--skills", "0.9,0.2"])

    out = capsys.readouterr().out
    assert code == 0
    assert "| skill | 档位 | 平均总分" in out
    assert "专家" in out and "实习生" in out