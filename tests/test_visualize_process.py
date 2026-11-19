"""过程可视化回归：示例轨迹数字锁定、奖励拆解可验算、HTML 自包含。"""

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "visualize_process", PROJECT_ROOT / "scripts" / "visualize_process.py")
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)


@pytest.fixture(scope="module")
def data():
    return vp.build_data(case_id="mock-mi-001", skill=0.9)


def test_example_trajectory_locked(data):
    trace = data["trace"]
    assert len(trace["turns"]) == 4
    assert trace["score"] == pytest.approx(1.0, abs=1e-4)
    assert trace["cost"] == pytest.approx(125, abs=0.01)
    assert trace["total_reward"] == pytest.approx(1.20, abs=1e-3)
    for key in ("diagnosis", "treatment", "safety"):
        assert trace["breakdown"][key] == pytest.approx(1.0, abs=1e-4)
    assert trace["breakdown"]["efficiency"] == pytest.approx(0.8, abs=1e-4)


def test_per_turn_rewards_verifiable(data):
    rewards = [turn["reward"] for turn in data["trace"]["turns"]]
    assert rewards == [pytest.approx(0.180, abs=1e-3),   # ASK: 0.2 新颖度 - 20*0.001
                       pytest.approx(0.100, abs=1e-3),   # EXAM: 0.2 - 100*0.001
                       pytest.approx(0.045, abs=1e-3),   # KNOWLEDGE: 0.05 - 5*0.001
                       pytest.approx(0.875, abs=1e-3)]   # FINAL: 1.0 - 125*0.001
    for turn in data["trace"]["turns"]:
        assert turn["reward_note"]


def test_trace_is_well_formed(data):
    turns = data["trace"]["turns"]
    assert turns[-1]["action"] == "FINAL"
    costs = [turn["cost"] for turn in turns]
    assert costs == sorted(costs), "累计成本必须单调不减"
    for turn in turns:
        assert turn["action"] in ("ASK", "EXAM", "KNOWLEDGE", "FINAL")
        assert -1.0 <= turn["reward"] <= 1.0
    assert data["trace"]["ground_truth"]["diagnosis"]


def test_html_is_self_contained(data):
    page = vp.render_html(data)
    assert "mock-mi-001" in page
    assert "Agent loop" in page and "多维反馈" in page
    assert "<svg" in page
    assert 'src="http' not in page and 'href="http' not in page
    assert len(page) > 8_000


def test_cli_writes_file(tmp_path):
    out = tmp_path / "process.html"
    assert vp.main(["--case-id", "mock-mi-001", "--skill", "0.9", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "mock-mi-001" in text and "多维反馈" in text