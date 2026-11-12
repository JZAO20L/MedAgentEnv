"""数据合成管线测试

覆盖：种子加载、标签均衡分片、JSON 容错解析、入库校验、dry-run 端到端与断点续跑。
"""

import json

import pytest

from src.medagent.data_pipeline import (
    PipelineConfig,
    SeedRecord,
    load_seeds,
    make_shards,
    normalize_case,
    parse_json_object,
    run_pipeline,
    validate_case,
)
from src.schema import MedicalCase


def _seed(seed_id: str, department: str = "内科") -> SeedRecord:
    return SeedRecord(
        seed_id=seed_id,
        department=department,
        disease="",
        narrative="患者近三日出现上腹隐痛，进食后加重，无发热。",
        difficulty="medium",
    )


def _valid_payload(**overrides):
    payload = {
        "case_id": "syn-001",
        "difficulty": "medium",
        "tags": ["消化内科"],
        "chief_complaint": "上腹隐痛三天，进食后加重。",
        "subjective": [
            {"keywords": ["疼痛部位", "腹痛位置"], "content": "疼痛位于上腹正中。", "necessity": True},
            {"keywords": ["进食关系"], "content": "进食后疼痛加重。", "necessity": True},
        ],
        "objective": [
            {"keywords": ["胃镜"], "content": "胃窦黏膜充血水肿，可见点状糜烂。", "necessity": True},
        ],
        "ground_truth": {
            "diagnosis": ["慢性胃炎"],
            "treatment": ["抑酸治疗", "胃黏膜保护剂"],
            "avoid": ["辛辣饮食"],
        },
    }
    payload.update(overrides)
    return payload


def test_load_seeds_from_directory(tmp_path):
    dept_dir = tmp_path / "内科"
    dept_dir.mkdir()
    (dept_dir / "cases.jsonl").write_text(
        json.dumps({"department": "内科", "disease": "胃炎", "text": "上腹痛三天。"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (dept_dir / "more.json").write_text(
        json.dumps([{"department": "内科", "chief_complaint": "咳嗽一周。"}], ensure_ascii=False),
        encoding="utf-8",
    )

    seeds = load_seeds(tmp_path)

    assert len(seeds) == 2
    assert all(seed.department == "内科" for seed in seeds)
    assert seeds[0].disease == "胃炎"


def test_load_seeds_skips_records_without_narrative(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"department": "内科", "text": "有主诉。"}, ensure_ascii=False),
            json.dumps({"department": "内科"}, ensure_ascii=False),
        ]),
        encoding="utf-8",
    )

    assert len(load_seeds(path)) == 1


def test_make_shards_balances_departments():
    seeds = [_seed(f"a{i}", "内科") for i in range(4)] + [_seed(f"b{i}", "外科") for i in range(4)]

    shards = make_shards(seeds, shard_size=4)

    assert len(shards) == 2
    for shard in shards:
        departments = {seed.department for seed in shard}
        assert departments == {"内科", "外科"}
    assert sum(len(shard) for shard in shards) == len(seeds)


def test_parse_json_object_handles_wrappers():
    fenced = '说明文字\n```json\n{"passed": true, "score": 5}\n```\n'
    prefixed = '结果如下：{"checks": {"schema": true}, "problems": []} 以上。'
    truncated = '{"passed": true, "score": 5'

    assert parse_json_object(fenced) == {"passed": True, "score": 5}
    assert parse_json_object(prefixed)["checks"]["schema"] is True
    assert parse_json_object(truncated) is None


def test_validate_case_flags_diagnosis_leak():
    payload = _valid_payload(chief_complaint="慢性胃炎复查，上腹不适。")
    case = MedicalCase.dict_to_case(payload)

    errors = validate_case(case)

    assert any("泄露诊断" in error for error in errors)


def test_validate_case_flags_duplicate_keywords():
    payload = _valid_payload()
    payload["objective"][0]["keywords"] = ["进食关系"]

    errors = validate_case(MedicalCase.dict_to_case(payload))

    assert any("keyword 重复" in error for error in errors)


def test_normalize_case_fills_metadata():
    case, errors = normalize_case(_valid_payload(case_id=""), _seed("seed-7", "消化内科"))

    assert errors == []
    assert case is not None
    assert case.case_id == "syn-seed-7"
    assert case.source == "synthetic"
    assert case.tags[0] == "消化内科"


@pytest.mark.parametrize("payload", [{}, {"chief_complaint": ""}, {"subjective": []}])
def test_normalize_case_rejects_broken_payload(payload):
    case, errors = normalize_case(payload, _seed("bad"))

    assert case is None
    assert errors


def test_dry_run_end_to_end_and_resume(tmp_path):
    seeds = [_seed(f"s{i}", "内科" if i % 2 else "外科") for i in range(5)]
    output = tmp_path / "cases.jsonl"
    checkpoint = tmp_path / "cases.jsonl.ckpt.jsonl"
    config = PipelineConfig(mode="quality", shard_size=2, max_workers=2, dry_run=True)

    stats = run_pipeline(seeds, config, output, checkpoint, resume=True)

    assert stats["processed"] == 5
    assert stats["passed"] == 5
    assert stats["pass_rate"] == 1.0
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 5

    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["source"] == "synthetic"
    assert record["synth_meta"]["mode"] == "quality"

    again = run_pipeline(seeds, config, output, checkpoint, resume=True)

    assert again["skipped"] == 5
    assert again["processed"] == 0
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_fast_mode_runs_generate_then_judge(tmp_path):
    config = PipelineConfig(mode="fast", shard_size=10, dry_run=True)
    stats = run_pipeline(
        [_seed("fast-1")],
        config,
        tmp_path / "fast.jsonl",
        tmp_path / "fast.ckpt.jsonl",
    )

    assert stats["passed"] == 1
    record = json.loads((tmp_path / "fast.ckpt.jsonl").read_text(encoding="utf-8").splitlines()[0])
    stages = [trace["stage"] for trace in record["traces"]]
    assert stages == ["generate", "judge"]


def test_strict_voting_rejects_when_not_enough_judges(tmp_path):
    config = PipelineConfig(mode="fast", shard_size=10, dry_run=True, min_passing_judges=2)
    stats = run_pipeline(
        [_seed("strict-1")],
        config,
        tmp_path / "strict.jsonl",
        tmp_path / "strict.ckpt.jsonl",
    )

    assert stats["passed"] == 0
    assert stats["rejected"] == 1
    assert (tmp_path / "strict.jsonl").read_text(encoding="utf-8") == ""