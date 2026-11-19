"""过程可视化回归：汇总数字与 sweep 同源、轨迹结构完整、HTML 自包含。"""

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
    return vp.build_data(case_id="mock-mi-001", skills=(1.0, 0.6, 0.2), n=6)


def test_summary_matches_locked_sweep_numbers(data):
    scores = [row["avg_score"] for row in data["summary"]]
    costs = [row["avg_cost"] for row in data["summary"]]
    assert scores == [pytest.approx(0.958, abs=5e-4),
                      pytest.approx(0.833, abs=5e-4),
                      pytest.approx(0.200, abs=5e-4)]
    assert costs == [pytest.approx(125, abs=0.5),
                     pytest.approx(142, abs=0.5),
                     pytest.approx(175, abs=0.5)]


def test_traces_are_well_formed(data):
    for trace in data["traces"]:
        turns = trace["turns"]
        assert turns, "轨迹不应为空"
        assert turns[-1]["action"] == "FINAL"
        costs = [turn["cost"] for turn in turns]
        assert costs == sorted(costs), "累计成本必须单调不减"
        for turn in turns:
            assert turn["action"] in ("ASK", "EXAM", "KNOWLEDGE", "FINAL")
            assert -1.0 <= turn["reward"] <= 1.0
        assert set(trace["breakdown"]) >= {"diagnosis", "treatment", "safety", "efficiency", "score"}


def test_html_is_self_contained(data):
    page = vp.render_html(data)
    assert "mock-mi-001" in page
    for tier in ("专家", "主治", "实习生"):
        assert tier in page
    assert "<svg" in page
    assert 'src="http' not in page and 'href="http' not in page
    assert len(page) > 10_000


def test_cli_writes_file(tmp_path):
    out = tmp_path / "process.html"
    assert vp.main(["--case-id", "mock-mi-001", "--skills", "1.0,0.2", "--n", "2",
                    "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "mock-mi-001" in text and "实习生" in text