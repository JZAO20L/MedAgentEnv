"""数据合成管线：Generate -> Review -> Rewrite -> Judge

对应设计文档 v0.5 §8.1，两种模式：

- ``fast``    : Generate -> Judge，2 次调用/样本，用于大规模迭代
- ``quality`` : Generate -> Review -> Rewrite -> Judge，4 次调用/样本

管线约束（v0.3 §4.3 / v0.5 §8）：

- 按科室轮询切片，保证每个 shard 的标签分布均衡
- checkpoint 逐条落盘，支持断点续跑
- shard 之间可配置暂停，规避 API 限流
- Judge 默认"全部裁判通过"才算通过（传 3 个裁判即 3/3）
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.schema import MedicalCase
from .llm import LLMClient, api_counter
from .version import __version__


# -----------------------------------------------------------------------------
# 角色配置
# -----------------------------------------------------------------------------


@dataclass
class RoleConfig:
    """一个模型角色的调用配置"""

    model: str
    base_url: str
    api_key_env: str = ""
    timeout: float = 180.0

    def build_client(self, max_tokens: int = 32768) -> LLMClient:
        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        return LLMClient(
            base_url=self.base_url,
            api_key=api_key or "EMPTY",
            model_name=self.model,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )


DEFAULT_ROLES: Dict[str, RoleConfig] = {
    "generator": RoleConfig(
        model="qwen3.5-plus",
        base_url="https://coding.dashscope.aliyuncs.com/v1",
        api_key_env="DASHSCOPE_API_KEY_CP",
    ),
    "reviewer": RoleConfig(
        model="glm-5",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_env="GLM_API_KEY",
    ),
    "rewriter": RoleConfig(
        model="qwen3.5-plus",
        base_url="https://coding.dashscope.aliyuncs.com/v1",
        api_key_env="DASHSCOPE_API_KEY_CP",
    ),
    "judge": RoleConfig(
        model="baichuan-m2",
        base_url="http://localhost:8200/v1",
        api_key_env="BAICHUAN_API_KEY",
    ),
}


@dataclass
class PipelineConfig:
    mode: str = "quality"                  # fast / quality
    shard_size: int = 500
    shard_interval: float = 0.0            # shard 间暂停秒数
    max_workers: int = 4
    temperature: float = 0.7
    judge_temperature: float = 0.1
    max_retries: int = 3
    generator: RoleConfig = field(default_factory=lambda: DEFAULT_ROLES["generator"])
    reviewer: RoleConfig = field(default_factory=lambda: DEFAULT_ROLES["reviewer"])
    rewriter: RoleConfig = field(default_factory=lambda: DEFAULT_ROLES["rewriter"])
    judges: List[RoleConfig] = field(default_factory=lambda: [DEFAULT_ROLES["judge"]])
    min_passing_judges: Optional[int] = None   # None -> 全部裁判都要通过
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("fast", "quality"):
            raise ValueError(f"未知模式: {self.mode}")
        if self.min_passing_judges is None:
            self.min_passing_judges = len(self.judges)
        if not self.judges:
            raise ValueError("至少需要一个 judge")


# -----------------------------------------------------------------------------
# 种子数据
# -----------------------------------------------------------------------------


@dataclass
class SeedRecord:
    seed_id: str
    department: str
    disease: str
    narrative: str
    difficulty: str = "medium"


_SEED_FIELD_ALIASES = {
    "department": ("department", "dept", "科室", "tag"),
    "disease": ("disease", "疾病", "diagnosis", "label"),
    "narrative": ("narrative", "text", "content", "description", "chief_complaint", "主诉", "case"),
}


def _pick(data: Dict[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _to_seed(data: Dict[str, Any], index: int) -> Optional[SeedRecord]:
    narrative = _pick(data, _SEED_FIELD_ALIASES["narrative"])
    if not narrative:
        return None

    tags = data.get("tags") if isinstance(data.get("tags"), list) else []
    department = _pick(data, _SEED_FIELD_ALIASES["department"]) or (str(tags[0]).strip() if tags else "未知科室")
    disease = _pick(data, _SEED_FIELD_ALIASES["disease"])
    difficulty = _pick(data, ("difficulty", "难度")) or "medium"
    seed_id = _pick(data, ("seed_id", "case_id", "id")) or f"seed-{index:06d}"

    return SeedRecord(
        seed_id=seed_id,
        department=department or "未知科室",
        disease=disease,
        narrative=narrative,
        difficulty=difficulty,
    )


def load_seeds(path: str | Path) -> List[SeedRecord]:
    """加载种子数据，支持单个 json/jsonl 或按科室分文件的目录"""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"种子数据不存在: {root}")

    if root.is_dir():
        files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".json", ".jsonl")])
    else:
        files = [root]

    seeds: List[SeedRecord] = []
    for file in files:
        for data in _iter_records(file):
            seed = _to_seed(data, len(seeds))
            if seed is not None:
                seeds.append(seed)
    return seeds


def _iter_records(file: Path) -> Iterable[Dict[str, Any]]:
    with open(file, "r", encoding="utf-8") as f:
        if file.suffix.lower() == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        else:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                return
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(payload, dict):
                # 兼容 {"data": [...]} 这类包装
                for value in payload.values():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        yield from value
                        break
                else:
                    yield payload


def make_shards(seeds: Sequence[SeedRecord], shard_size: int) -> List[List[SeedRecord]]:
    """按科室轮询切片，保证每个 shard 的标签分布均衡"""
    if shard_size <= 0:
        raise ValueError("shard_size 必须为正数")

    buckets: "OrderedDict[str, deque]" = OrderedDict()
    for seed in seeds:
        buckets.setdefault(seed.department, deque()).append(seed)

    shards: List[List[SeedRecord]] = []
    current: List[SeedRecord] = []
    while any(buckets.values()):
        for queue in buckets.values():
            if not queue:
                continue
            current.append(queue.popleft())
            if len(current) >= shard_size:
                shards.append(current)
                current = []
    if current:
        shards.append(current)
    return shards


# -----------------------------------------------------------------------------
# JSON 解析与校验
# -----------------------------------------------------------------------------


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取 JSON 对象，容忍代码块、前后缀与截断"""
    if not text or not text.strip():
        return None

    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped):
        try:
            payload = json.loads(block.strip())
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            payload = json.loads(stripped[start : idx + 1])
                            if isinstance(payload, dict):
                                return payload
                        except json.JSONDecodeError:
                            pass
                        break
        start = stripped.find("{", start + 1)

    return None


def validate_case(case: MedicalCase) -> List[str]:
    """校验合成病例是否可入库，返回错误列表"""
    errors: List[str] = []

    if len(case.chief_complaint.strip()) < 5:
        errors.append("主诉过短")
    if not (2 <= len(case.subjective) <= 10):
        errors.append(f"subjective 条数异常: {len(case.subjective)}")
    if not (1 <= len(case.objective) <= 10):
        errors.append(f"objective 条数异常: {len(case.objective)}")

    seen_keywords: Dict[str, str] = {}
    for kind, items in (("subjective", case.subjective), ("objective", case.objective)):
        for idx, item in enumerate(items):
            if not item.keywords:
                errors.append(f"{kind}[{idx}] 缺少 keywords")
            if len(item.content.strip()) < 5:
                errors.append(f"{kind}[{idx}] content 过短")
            for keyword in item.keywords:
                if len(keyword) > 20:
                    errors.append(f"{kind}[{idx}] keyword 过长: {keyword}")
                if keyword in seen_keywords:
                    errors.append(f"keyword 重复: {keyword}")
                seen_keywords[keyword] = f"{kind}[{idx}]"

    if not case.ground_truth.diagnosis:
        errors.append("ground_truth.diagnosis 为空")
    if not case.ground_truth.treatment:
        errors.append("ground_truth.treatment 为空")

    for diagnosis in case.ground_truth.diagnosis:
        if diagnosis and diagnosis in case.chief_complaint:
            errors.append(f"主诉泄露诊断: {diagnosis}")

    return errors


def normalize_case(raw: Dict[str, Any], seed: SeedRecord) -> Tuple[Optional[MedicalCase], List[str]]:
    """把模型输出规整为 MedicalCase，并做入库前校验"""
    if not isinstance(raw, dict):
        return None, ["模型输出不是 JSON 对象"]

    payload = dict(raw)
    if not payload.get("case_id"):
        payload["case_id"] = f"syn-{seed.seed_id}"
    if not payload.get("difficulty"):
        payload["difficulty"] = seed.difficulty
    payload["source"] = "synthetic"

    tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    if seed.department and seed.department not in tags:
        tags.insert(0, seed.department)
    payload["tags"] = tags

    if not payload.get("chief_complaint"):
        return None, ["缺少 chief_complaint"]

    case = MedicalCase.dict_to_case(payload)
    errors = validate_case(case)
    if errors:
        return None, errors
    return case, []


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

CASE_SCHEMA_HINT = json.dumps(
    {
        "case_id": "syn-000001",
        "difficulty": "easy|medium|hard",
        "tags": ["内科", "心血管"],
        "chief_complaint": "患者自述……（不得出现诊断名）",
        "subjective": [
            {"keywords": ["疼痛性质", "胸痛特点"], "content": "压榨样疼痛，向左肩放射", "necessity": True}
        ],
        "objective": [
            {"keywords": ["心电图", "ECG"], "content": "V1-V4 导联 ST 段抬高 0.3mV", "necessity": True}
        ],
        "ground_truth": {
            "diagnosis": ["急性前壁心肌梗死"],
            "treatment": ["急诊 PCI", "双联抗血小板"],
            "avoid": ["运动负荷试验"],
        },
    },
    ensure_ascii=False,
    indent=2,
)


def _generate_prompt(seed: SeedRecord) -> str:
    return f"""你是一名三甲医院的主治医师，正在为"AI 问诊 Agent 评测集"编写一份模拟病例。

## 种子信息
- 科室：{seed.department}
- 疾病（可能为空，为空时由你根据主诉判断）：{seed.disease or "未指定"}
- 难度：{seed.difficulty}
- 原始素材：{seed.narrative}

## 编写要求
1. chief_complaint 是患者自己的话，1-3 句，只写症状与病程，绝对不能出现最终诊断名称。
2. subjective 写 3-8 条"只有问诊才会披露"的信息（现病史、既往史、用药史、过敏史、家族史、伴随症状）。
3. objective 写 2-6 条检查/化验/影像结果。
4. 每条信息给 1-4 个 keywords：必须是医生自然会用来提问或开检查的说法（如"疼痛性质""心电图""肌钙蛋白"），不能是整句，也不能含诊断名。Agent 只有命中 keywords 才能拿到 content。
5. necessity 表示该条信息是否为做出正确诊断所必需。
6. ground_truth.diagnosis 给 1-3 个规范诊断名；treatment 给 2-5 条具体处置（药物/手术/支持治疗），不要写"去医院"；avoid 给 0-3 条该病例下的禁忌处置。
7. 全部信息必须自洽：客观检查要能支撑诊断，不得出现互相矛盾的结果。

## 输出
只输出一个 JSON 对象，不要解释，不要 Markdown 代码块。字段结构：
{CASE_SCHEMA_HINT}
"""


def _review_prompt(case: MedicalCase) -> str:
    return f"""你是独立的复核模型，负责在入库前挑错。请检查下面这份合成病例：

{json.dumps(case.case_to_dict(), ensure_ascii=False, indent=2)}

逐项检查：
1. 主诉是否泄露诊断名；
2. keywords 是否是医生自然会说的词，能否被问诊/检查动作命中；
3. subjective 与 objective 披露的信息是否足以支撑 ground_truth；
4. ground_truth 的 diagnosis / treatment / avoid 是否符合临床常规；
5. 是否存在前后矛盾（例如化验正常却诊断细菌感染）。

只输出 JSON：
{{"needs_rewrite": true/false, "severity": "none|minor|major", "issues": ["具体问题，指明字段"]}}
"""


def _rewrite_prompt(case: MedicalCase, review: Dict[str, Any]) -> str:
    return f"""请根据复核意见修订下面这份病例，保持 case_id 不变，其余字段可改。

## 原病例
{json.dumps(case.case_to_dict(), ensure_ascii=False, indent=2)}

## 复核意见
{json.dumps(review, ensure_ascii=False, indent=2)}

要求：只修被指出的问题，不要新增无关内容；修订后仍然满足原始编写要求。
只输出修订后的完整 JSON 对象，字段结构：
{CASE_SCHEMA_HINT}
"""


def _judge_prompt(case: MedicalCase) -> str:
    return f"""你是医疗数据终审专家，请判断这份合成病例能否进入评测集。

{json.dumps(case.case_to_dict(), ensure_ascii=False, indent=2)}

检查项：
- schema：字段完整，keywords 非空，diagnosis / treatment 非空；
- decidable：仅凭 keywords 能披露的信息足以推出 ground_truth；
- no_leak：主诉与 keywords 不含诊断名；
- gt_correct：诊断与处置符合临床规范，avoid 合理。

只输出 JSON：
{{"passed": true/false, "score": 1-5, "problems": ["..."], "checks": {{"schema": true, "decidable": true, "no_leak": true, "gt_correct": true}}}}

判定规则：checks 四项全为 true 且 score >= 4 时 passed 才为 true。
"""


# -----------------------------------------------------------------------------
# 离线模式（无 API Key 时验证管线）
# -----------------------------------------------------------------------------


def _mock_response(stage: str, context: Dict[str, Any]) -> str:
    """dry-run 下的确定性响应，仅用于跑通管线与测试"""
    if stage == "generate":
        seed: SeedRecord = context["seed"]
        payload = {
            "case_id": f"syn-{seed.seed_id}",
            "difficulty": seed.difficulty,
            "tags": [seed.department],
            "chief_complaint": seed.narrative[:80],
            "subjective": [
                {"keywords": ["症状特点", "起病情况"], "content": "症状于近日出现，逐渐加重。", "necessity": True},
                {"keywords": ["既往史", "病史"], "content": "既往有慢性病史，长期服药。", "necessity": True},
                {"keywords": ["伴随症状"], "content": "伴轻度乏力，无发热。", "necessity": False},
            ],
            "objective": [
                {"keywords": ["血常规", "化验"], "content": "白细胞计数轻度升高。", "necessity": True},
                {"keywords": ["影像学检查", "CT"], "content": "影像未见明显异常。", "necessity": False},
            ],
            "ground_truth": {
                "diagnosis": [seed.disease or "待明确诊断"],
                "treatment": ["对症支持治疗", "定期随访"],
                "avoid": ["剧烈运动"],
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    if stage == "review":
        return json.dumps({"needs_rewrite": False, "severity": "none", "issues": []}, ensure_ascii=False)

    if stage == "rewrite":
        case: MedicalCase = context["case"]
        return json.dumps(case.case_to_dict(), ensure_ascii=False)

    if stage == "judge":
        return json.dumps(
            {
                "passed": True,
                "score": 4,
                "problems": [],
                "checks": {"schema": True, "decidable": True, "no_leak": True, "gt_correct": True},
            },
            ensure_ascii=False,
        )

    raise ValueError(f"未知阶段: {stage}")


# -----------------------------------------------------------------------------
# 合成器
# -----------------------------------------------------------------------------


@dataclass
class StageTrace:
    stage: str
    ok: bool
    latency: float = 0.0
    detail: str = ""
    error: str = ""


@dataclass
class PipelineRecord:
    seed_id: str
    status: str                       # passed / rejected / failed
    case: Optional[Dict[str, Any]] = None
    traces: List[StageTrace] = field(default_factory=list)
    judge_votes: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "status": self.status,
            "case": self.case,
            "traces": [trace.__dict__ for trace in self.traces],
            "judge_votes": self.judge_votes,
            "reason": self.reason,
        }


class CaseSynthesizer:
    """单样本的 4 阶段合成器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._clients: Dict[Tuple[str, str, str], LLMClient] = {}

    # -- 模型调用 ---------------------------------------------------------

    def _client(self, role: RoleConfig) -> LLMClient:
        key = (role.model, role.base_url, role.api_key_env)
        if key not in self._clients:
            self._clients[key] = role.build_client()
        return self._clients[key]

    def _call(self, role: RoleConfig, prompt: str, stage: str, context: Dict[str, Any],
              temperature: float) -> str:
        if self.config.dry_run:
            return _mock_response(stage, context)

        client = self._client(role)
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                response = client.call(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=32768,
                    response_format={"type": "json_object"},
                )
                if isinstance(response, dict):
                    return json.dumps(response, ensure_ascii=False)
                return response or ""
            except Exception as exc:  # noqa: BLE001 - 网络/限流错误统一重试
                last_error = exc
                message = str(exc).lower()
                wait = 30 * (attempt + 1) if ("429" in message or "rate" in message) else 2 ** attempt
                if attempt < self.config.max_retries - 1:
                    print(f"[pipeline] {role.model} 调用失败，{wait}s 后重试：{exc}")
                    time.sleep(wait)
        raise RuntimeError(f"{role.model} 调用失败: {last_error}")

    # -- 各阶段 -----------------------------------------------------------

    def generate(self, seed: SeedRecord) -> Tuple[Optional[MedicalCase], StageTrace]:
        started = time.time()
        try:
            text = self._call(
                self.config.generator, _generate_prompt(seed), "generate",
                {"seed": seed}, self.config.temperature,
            )
            raw = parse_json_object(text)
            if raw is None:
                return None, StageTrace("generate", False, time.time() - started, error="JSON 解析失败")
            case, errors = normalize_case(raw, seed)
            if case is None:
                return None, StageTrace("generate", False, time.time() - started,
                                        detail="; ".join(errors), error="校验未通过")
            return case, StageTrace("generate", True, time.time() - started)
        except Exception as exc:  # noqa: BLE001
            return None, StageTrace("generate", False, time.time() - started, error=str(exc))

    def review(self, case: MedicalCase) -> Tuple[Optional[Dict[str, Any]], StageTrace]:
        started = time.time()
        try:
            text = self._call(
                self.config.reviewer, _review_prompt(case), "review",
                {"case": case}, self.config.temperature,
            )
            payload = parse_json_object(text)
            if payload is None:
                return None, StageTrace("review", False, time.time() - started, error="JSON 解析失败")
            return payload, StageTrace("review", True, time.time() - started)
        except Exception as exc:  # noqa: BLE001
            return None, StageTrace("review", False, time.time() - started, error=str(exc))

    def rewrite(self, case: MedicalCase, review: Dict[str, Any],
                seed: SeedRecord) -> Tuple[Optional[MedicalCase], StageTrace]:
        started = time.time()
        try:
            text = self._call(
                self.config.rewriter, _rewrite_prompt(case, review), "rewrite",
                {"case": case}, self.config.temperature,
            )
            raw = parse_json_object(text)
            if raw is None:
                return None, StageTrace("rewrite", False, time.time() - started, error="JSON 解析失败")
            revised, errors = normalize_case(raw, seed)
            if revised is None:
                return None, StageTrace("rewrite", False, time.time() - started,
                                        detail="; ".join(errors), error="修订稿校验未通过")
            revised.case_id = case.case_id
            return revised, StageTrace("rewrite", True, time.time() - started)
        except Exception as exc:  # noqa: BLE001
            return None, StageTrace("rewrite", False, time.time() - started, error=str(exc))

    def judge(self, case: MedicalCase) -> Tuple[List[Dict[str, Any]], StageTrace]:
        started = time.time()
        votes: List[Dict[str, Any]] = []
        failures: List[str] = []

        for index, role in enumerate(self.config.judges):
            try:
                text = self._call(
                    role, _judge_prompt(case), "judge", {"case": case},
                    self.config.judge_temperature,
                )
                payload = parse_json_object(text)
                if payload is None:
                    failures.append(f"judge[{index}] JSON 解析失败")
                    continue
                checks = payload.get("checks") or {}
                passed = bool(payload.get("passed")) and bool(checks) and all(
                    bool(checks.get(name)) for name in ("schema", "decidable", "no_leak", "gt_correct")
                ) and float(payload.get("score", 0)) >= 4
                votes.append({
                    "model": role.model,
                    "passed": passed,
                    "score": payload.get("score"),
                    "problems": payload.get("problems", []),
                })
            except Exception as exc:  # noqa: BLE001
                failures.append(f"judge[{index}] {exc}")

        if not votes:
            return [], StageTrace("judge", False, time.time() - started, error="; ".join(failures))

        passing = sum(1 for vote in votes if vote["passed"])
        trace = StageTrace(
            "judge",
            passing >= self.config.min_passing_judges,
            time.time() - started,
            detail=f"{passing}/{len(votes)} 通过（要求 {self.config.min_passing_judges}）",
        )
        return votes, trace

    # -- 单样本流程 -------------------------------------------------------

    def run(self, seed: SeedRecord) -> PipelineRecord:
        record = PipelineRecord(seed_id=seed.seed_id, status="failed")

        case, trace = self.generate(seed)
        record.traces.append(trace)
        if case is None:
            record.reason = f"generate: {trace.error or trace.detail}"
            return record

        if self.config.mode == "quality":
            review, review_trace = self.review(case)
            record.traces.append(review_trace)
            if review and review.get("needs_rewrite"):
                rewritten, rewrite_trace = self.rewrite(case, review, seed)
                record.traces.append(rewrite_trace)
                if rewritten is not None:
                    case = rewritten

        votes, judge_trace = self.judge(case)
        record.traces.append(judge_trace)
        record.judge_votes = votes

        if not judge_trace.ok:
            record.status = "rejected"
            record.reason = judge_trace.error or judge_trace.detail
            return record

        record.status = "passed"
        record.case = self._finalize(case, seed)
        return record

    def _finalize(self, case: MedicalCase, seed: SeedRecord) -> Dict[str, Any]:
        payload = case.case_to_dict()
        payload["synth_meta"] = {
            "pipeline_version": __version__,
            "mode": self.config.mode,
            "seed_id": seed.seed_id,
            "seed_department": seed.department,
            "generator": self.config.generator.model,
            "judges": [role.model for role in self.config.judges],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return payload


# -----------------------------------------------------------------------------
# 批量运行
# -----------------------------------------------------------------------------


def _load_done(checkpoint: Path) -> set:
    done = set()
    if not checkpoint.exists():
        return done
    with open(checkpoint, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "passed":
                done.add(record.get("seed_id"))
    return done


def run_pipeline(
    seeds: Sequence[SeedRecord],
    config: PipelineConfig,
    output_path: str | Path,
    checkpoint_path: str | Path,
    resume: bool = True,
) -> Dict[str, Any]:
    """跑完整管线，返回统计信息"""
    output = Path(output_path)
    checkpoint = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done(checkpoint) if resume else set()
    pending = [seed for seed in seeds if seed.seed_id not in done]
    skipped = len(seeds) - len(pending)

    stats = {
        "total": len(seeds),
        "skipped": skipped,
        "processed": 0,
        "passed": 0,
        "rejected": 0,
        "failed": 0,
        "api_calls_before": api_counter.get_count(),
    }
    if not pending:
        stats["pass_rate"] = 0.0
        stats["api_calls"] = 0
        return stats

    synthesizer = CaseSynthesizer(config)
    shards = make_shards(pending, config.shard_size)
    started = time.time()

    with open(checkpoint, "a", encoding="utf-8") as ckpt, open(output, "a", encoding="utf-8") as out:
        for shard_index, shard in enumerate(shards, start=1):
            print(f"[pipeline] shard {shard_index}/{len(shards)}：{len(shard)} 条")
            with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
                futures = {pool.submit(synthesizer.run, seed): seed for seed in shard}
                for future in as_completed(futures):
                    record = future.result()
                    ckpt.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    ckpt.flush()

                    stats["processed"] += 1
                    stats[record.status] = stats.get(record.status, 0) + 1

                    if record.status == "passed" and record.case:
                        out.write(json.dumps(record.case, ensure_ascii=False) + "\n")
                        out.flush()

                    if stats["processed"] % 10 == 0 or record.status != "passed":
                        print(
                            f"[pipeline] {stats['processed']}/{len(pending)} "
                            f"passed={stats['passed']} rejected={stats['rejected']} failed={stats['failed']}"
                        )

            if shard_index < len(shards) and config.shard_interval > 0:
                print(f"[pipeline] 暂停 {config.shard_interval}s 以规避限流")
                time.sleep(config.shard_interval)

    elapsed = time.time() - started
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["pass_rate"] = stats["passed"] / stats["processed"] if stats["processed"] else 0.0
    stats["api_calls"] = api_counter.get_count() - stats["api_calls_before"]
    stats["output"] = str(output)
    stats["checkpoint"] = str(checkpoint)
    return stats