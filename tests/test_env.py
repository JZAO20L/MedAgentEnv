"""环境与脚本化 agent 测试"""

import asyncio

import pytest

from src.medagent.data_pipeline import validate_case
from src.medagent.demo_agent import run_episode
from src.medagent.env import CaseCatalog, MedAgentEnv, parse_action, split_keywords


def _env(case_id="mock-mi-001", max_steps=16):
    return MedAgentEnv(env_config={"case_id": case_id, "max_steps": max_steps})


def test_mock_cases_pass_validation():
    catalog = CaseCatalog()
    assert len(catalog.cases) >= 6
    for case in catalog.cases:
        assert validate_case(case) == [], case.case_id


def test_reset_returns_complaint_and_prompt():
    env = _env()
    observation, info, system_message = asyncio.run(env.reset())

    assert "胸痛" in observation
    assert info["case_id"] == "mock-mi-001"
    assert "<action>ASK</action>" in system_message


def test_parse_action_supports_xml_json_and_bare():
    xml = "<act><action>ASK</action><keywords>疼痛性质, 持续时间</keywords></act>"
    payload = '{"action": "exam", "keywords": ["心电图"]}'
    bare = "FINAL: 急性前壁心肌梗死"

    assert parse_action(xml)["keywords"] == ["疼痛性质", "持续时间"]
    assert parse_action(payload)["action"] == "EXAM"
    assert parse_action(bare)["action"] == "FINAL"


def test_split_keywords_handles_chinese_punctuation():
    assert split_keywords("疼痛性质，持续时间、心电图") == ["疼痛性质", "持续时间", "心电图"]


def test_ask_novelty_then_repeat_is_penalised():
    env = _env()

    async def scenario():
        await env.reset()
        _, first, done, _ = await env.step(
            "<act><action>ASK</action><keywords>疼痛性质</keywords></act>")
        _, second, done, _ = await env.step(
            "<act><action>ASK</action><keywords>疼痛性质</keywords></act>")
        return first, second, done

    first, second, done = asyncio.run(scenario())

    assert first > 0
    assert second < 0
    assert done is False


def test_exam_costs_more_than_ask():
    env = _env()

    async def scenario():
        await env.reset()
        await env.step("<act><action>ASK</action><keywords>疼痛性质</keywords></act>")
        ask_cost = env.state.cost
        await env.step("<act><action>EXAM</action><keywords>心电图</keywords></act>")
        return ask_cost, env.state.cost

    ask_cost, total = asyncio.run(scenario())

    assert ask_cost > 0
    assert total - ask_cost > ask_cost


def test_final_scores_full_marks_for_correct_answer():
    env = _env()

    async def scenario():
        await env.reset()
        return await env.step(
            "<act><action>FINAL</action>"
            "<diagnosis>急性前壁心肌梗死</diagnosis>"
            "<treat>急诊PCI；双联抗血小板治疗；抗凝治疗</treat></act>")

    _, reward, done, info = asyncio.run(scenario())

    assert done is True
    assert info["breakdown"]["diagnosis"] == 1.0
    assert info["breakdown"]["treatment"] == 1.0
    assert info["breakdown"]["safety"] == 1.0
    assert reward == pytest.approx(1.0)


def test_final_with_wrong_answer_scores_low():
    env = _env()

    async def scenario():
        await env.reset()
        return await env.step(
            "<act><action>FINAL</action><diagnosis>胃炎</diagnosis><treat>观察</treat></act>")

    _, _, _, info = asyncio.run(scenario())

    assert info["breakdown"]["diagnosis"] == 0.0
    assert info["breakdown"]["score"] < 0.5


def test_safety_violation_zeroes_safety_score():
    env = _env()

    async def scenario():
        await env.reset()
        return await env.step(
            "<act><action>FINAL</action>"
            "<diagnosis>急性前壁心肌梗死</diagnosis>"
            "<treat>运动负荷试验评估缺血</treat></act>")

    _, _, _, info = asyncio.run(scenario())

    assert info["breakdown"]["safety"] == 0.0


def test_invalid_action_penalty():
    env = _env()

    async def scenario():
        await env.reset()
        return await env.step("我觉得应该先问问病人。")

    _, reward, done, info = asyncio.run(scenario())

    assert reward < 0
    assert done is False
    assert info["error"] == "invalid_action"


def test_max_steps_terminates_episode():
    env = _env(max_steps=1)

    async def scenario():
        await env.reset()
        return await env.step("<act><action>ASK</action><keywords>疼痛性质</keywords></act>")

    _, _, done, info = asyncio.run(scenario())

    assert done is True
    assert info["done_reason"] == "max_steps"
    assert env.state.breakdown["score"] == 0.0


def test_scripted_episode_reaches_high_score():
    result = asyncio.run(run_episode(_env(), skill=0.9))

    assert result["done_reason"] == "final"
    assert result["score"] >= 0.9
    assert result["cost"] > 0
    assert 3 <= result["turns"] <= 8

    actions = [message["content"] for message in result["transcript"] if message["role"] == "assistant"]
    assert any("ASK" in action for action in actions)
    assert any("FINAL" in action for action in actions)


def test_low_skill_scores_lower_than_high_skill():
    good = asyncio.run(run_episode(_env(), skill=1.0))
    bad = asyncio.run(run_episode(_env(), skill=0.2))

    assert bad["score"] < good["score"]