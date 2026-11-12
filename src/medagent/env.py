"""OpenEnv 风格的医疗问诊环境

对外接口与 ms-swift ``Env`` 一致（async reset/step/close）：

    observation, info, system_message = await env.reset(config)
    next_observation, reward, done, info = await env.step(messages)

环境本身不依赖 swift，可单独用于 CLI / WebUI / 单元测试；
swift 侧通过 ``scripts/swift_plugin.py`` 把本类注册进 ``envs``。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import MedicalCase

from .rewards import (
    ACTION_COST,
    COST_SCALE,
    clip_step_reward,
    efficiency_reward,
    terminal_breakdown,
)

MOCK_DIR = PROJECT_ROOT / "assets" / "mock"
SYSTEM_PROMPT = """/no_think
你是一名在模拟医院里工作的临床医生。你只能通过下面四种动作获取信息或结束问诊。

## 动作格式（每次只输出一个动作）

问诊（了解症状、病史）：
<act>
<action>ASK</action>
<keywords>疼痛性质, 持续时间</keywords>
</act>

检查（开化验、影像）：
<act>
<action>EXAM</action>
<keywords>心电图, 心肌酶</keywords>
</act>

查知识库（诊断标准、治疗指南）：
<act>
<action>KNOWLEDGE</action>
<keywords>心肌梗死, 诊断标准</keywords>
</act>

结束并给出结论：
<act>
<action>FINAL</action>
<diagnosis>你的诊断</diagnosis>
<treat>具体处置方案</treat>
</act>

## 规则
1. keywords 要用医生自然会说的词，逗号分隔，不要写整句。
2. 重复问同一件事不会得到新信息，还会有额外成本。
3. 每开一项检查都会计入诊疗成本，请避免不必要的检查。
4. 信息足够时尽早 FINAL；超过轮次上限会自动结束。
5. 处置方案不能包含该病例的禁忌操作。
"""


# -----------------------------------------------------------------------------
# 数据加载
# -----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class CaseCatalog:
    """mock 病例目录（可替换为真实数据集）"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else MOCK_DIR / "cases.jsonl"
        self.cases: List[MedicalCase] = []
        if self.path.exists():
            for record in _read_jsonl(self.path):
                case = MedicalCase.dict_to_case(record)
                case.knowledge_keywords = record.get("knowledge_keywords", [])
                self.cases.append(case)
        self._by_id = {case.case_id: case for case in self.cases}

    def get(self, case_id: Optional[str] = None) -> MedicalCase:
        if case_id and case_id in self._by_id:
            return self._by_id[case_id]
        if not self.cases:
            raise RuntimeError(f"病例目录为空: {self.path}")
        return self.cases[0]

    def list(self) -> List[Dict[str, str]]:
        return [
            {
                "case_id": case.case_id,
                "chief_complaint": case.chief_complaint,
                "department": case.tags[0] if case.tags else "未知",
                "difficulty": case.difficulty,
            }
            for case in self.cases
        ]


class MockKnowledgeBase:
    """小知识库：按关键词命中数排序，替代 371k 的 ResponseMed.json"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else MOCK_DIR / "knowledge.json"
        self.records: List[Dict[str, str]] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    instruction = item.get("instruction", "")
                    output = item.get("output", "")
                    self.records.append({
                        "instruction": instruction,
                        "output": output,
                        "full_text": f"{instruction} {output}".lower(),
                    })

    def search(self, keywords: Sequence[str], top_k: int = 3) -> List[Dict[str, str]]:
        scored = []
        for index, record in enumerate(self.records):
            score = sum(1 for keyword in keywords if keyword and keyword.lower() in record["full_text"])
            if score:
                scored.append((score, index, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:top_k]]

    def format_results(self, results: Sequence[Dict[str, str]], max_chars: int = 600) -> str:
        if not results:
            return "知识库未检索到相关内容。"
        blocks = [f"{index + 1}. {item['instruction']}\n{item['output']}" for index, item in enumerate(results)]
        text = "\n\n".join(blocks)
        return text[:max_chars]


# -----------------------------------------------------------------------------
# 动作解析
# -----------------------------------------------------------------------------


def split_keywords(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,，、;；\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def parse_action(text: str) -> Dict[str, Any]:
    """兼容 XML 动作块、JSON 对象、`ASK: a,b` 三种写法"""
    result: Dict[str, Any] = {"action": "", "keywords": [], "diagnosis": "", "treatment": ""}
    if not text:
        return result

    lowered = text.lower()

    if "<act>" in lowered:
        def tag(name: str) -> str:
            match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        result["action"] = tag("action").upper()
        result["keywords"] = split_keywords(tag("keywords"))
        result["diagnosis"] = tag("diagnosis")
        result["treatment"] = tag("treat") or tag("treatment")
        return result

    payload = None
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict) and payload.get("action"):
        result["action"] = str(payload.get("action", "")).upper()
        result["keywords"] = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()]
        result["diagnosis"] = str(payload.get("diagnosis", ""))
        result["treatment"] = str(payload.get("treatment", ""))
        return result

    match = re.match(r"^\s*(ASK|EXAM|KNOWLEDGE|FINAL)\s*[:：]\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if match:
        result["action"] = match.group(1).upper()
        body = match.group(2).strip()
        if result["action"] == "FINAL":
            result["diagnosis"] = body
        else:
            result["keywords"] = split_keywords(body)
    return result


def last_assistant_text(action: Any) -> str:
    """把 swift 的 Messages / 单条消息 / 纯文本统一成最后一段 assistant 文本"""
    if action is None:
        return ""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        return str(action.get("content", ""))
    if isinstance(action, (list, tuple)):
        for message in reversed(action):
            if isinstance(message, dict) and message.get("role") in ("assistant", None):
                content = message.get("content")
                if content:
                    return str(content)
        for message in reversed(action):
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])
    return str(action)


def _merge_config(config: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(fallback or {})
    if config is None:
        return merged
    if isinstance(config, dict):
        merged.update(config)
        return merged
    data_dict = getattr(config, "data_dict", None)
    if isinstance(data_dict, dict):
        merged.update(data_dict)
        for key in ("env_config",):
            nested = data_dict.get(key)
            if isinstance(nested, dict):
                merged.update(nested)
    return merged


# -----------------------------------------------------------------------------
# 环境状态
# -----------------------------------------------------------------------------


@dataclass
class EpisodeState:
    case: MedicalCase
    max_steps: int = 16
    turn: int = 0
    done: bool = False
    done_reason: str = ""
    disclosed_subjective: set = field(default_factory=set)
    disclosed_objective: set = field(default_factory=set)
    knowledge_queries: List[Dict[str, Any]] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=lambda: {"ASK": 0, "EXAM": 0, "KNOWLEDGE": 0})
    cost: float = 0.0
    total_reward: float = 0.0
    diagnosis: str = ""
    treatment: str = ""
    breakdown: Dict[str, float] = field(default_factory=dict)
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "turn": self.turn,
            "max_steps": self.max_steps,
            "done": self.done,
            "done_reason": self.done_reason,
            "cost": round(self.cost, 2),
            "total_reward": round(self.total_reward, 4),
            "action_counts": dict(self.action_counts),
            "diagnosis": self.diagnosis,
            "treatment": self.treatment,
            "breakdown": {key: round(value, 4) for key, value in self.breakdown.items()},
            "ground_truth": {
                "diagnosis": self.case.ground_truth.diagnosis,
                "treatment": self.case.ground_truth.treatment,
                "avoid": self.case.ground_truth.avoid,
            },
        }


# -----------------------------------------------------------------------------
# 环境
# -----------------------------------------------------------------------------


class MedAgentEnv:
    """单病例的交互式环境，一个实例对应一条轨迹"""

    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        catalog: Optional[CaseCatalog] = None,
        knowledge_base: Optional[MockKnowledgeBase] = None,
    ):
        self.env_config = dict(env_config or {})
        self.catalog = catalog or CaseCatalog()
        self.knowledge_base = knowledge_base or MockKnowledgeBase()
        self.state: Optional[EpisodeState] = None

    # -- 生命周期 ---------------------------------------------------------

    async def reset(self, config: Any = None) -> Tuple[str, Dict[str, Any], str]:
        merged = _merge_config(config, self.env_config)
        case = self.catalog.get(merged.get("case_id"))
        max_steps = int(merged.get("max_steps", 16))

        self.state = EpisodeState(case=case, max_steps=max_steps)
        self.state.transcript.append({"role": "system", "content": SYSTEM_PROMPT})

        observation = (
            f"【病例】主诉：{case.chief_complaint}\n"
            f"【科室】{case.tags[0] if case.tags else '未知'}\n"
            f"【规则】最多 {max_steps} 轮；请用 <act> 输出 ASK / EXAM / KNOWLEDGE / FINAL 之一。"
        )
        info = {"case_id": case.case_id, "max_steps": max_steps}
        return observation, info, SYSTEM_PROMPT

    async def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("请先调用 reset()")
        if self.state.done:
            return "", 0.0, True, {"error": "episode already done"}

        state = self.state
        text = last_assistant_text(action)
        parsed = parse_action(text)
        parsed_action = parsed["action"]
        state.turn += 1

        info: Dict[str, Any] = {
            "turn": state.turn,
            "action": parsed_action,
            "keywords": parsed["keywords"],
        }

        if parsed_action in ("ASK", "EXAM", "KNOWLEDGE"):
            observation, reward = self._handle_info_action(parsed, info)
        elif parsed_action == "FINAL":
            observation, reward = self._handle_final(parsed, info)
        else:
            reward = -0.2
            observation = (
                "无法解析动作。请严格按 <act><action>ASK|EXAM|KNOWLEDGE|FINAL</action>...</act> 输出，"
                "且一次只输出一个动作。"
            )
            info.update({"error": "invalid_action", "step_reward": reward})
            state.total_reward += reward

        if not state.done and state.turn >= state.max_steps:
            self._finalize(reason="max_steps", diagnosis=state.diagnosis, treatment=state.treatment)
            observation += "\n\n已达到轮次上限，问诊结束。"
            reward += 0.0
            info["done_reason"] = "max_steps"

        state.transcript.append({"role": "assistant", "content": text, "reward": round(reward, 4)})
        state.transcript.append({"role": "user", "content": observation})

        info["step_reward"] = round(reward, 4)
        info["total_reward"] = round(state.total_reward, 4)
        info["cost"] = round(state.cost, 2)
        info["done"] = state.done
        return observation, float(reward), state.done, info

    async def close(self) -> None:
        if self.state is not None:
            self.state.done = True
            self.state.done_reason = self.state.done_reason or "closed"

    # -- 动作实现 ---------------------------------------------------------

    def _handle_info_action(self, parsed: Dict[str, Any], info: Dict[str, Any]) -> Tuple[str, float]:
        state = self.state
        action = parsed["action"]
        keywords = parsed["keywords"] or []

        if not keywords:
            state.total_reward -= 0.1
            return "请提供 keywords，例如 <keywords>疼痛性质, 持续时间</keywords>。", -0.1

        if action == "ASK":
            items = state.case.subjective
            matched = self._match_items(items, keywords)
            new = [index for index in matched if index not in state.disclosed_subjective]
            state.disclosed_subjective.update(new)
            cost = ACTION_COST["ASK"] * max(1, len(keywords))
            body = "\n".join(f"- {items[index].content}" for index in matched) or "患者表示不清楚这个问题。"
            observation = f"【问诊结果】\n{body}"
            novelty = min(0.3, 0.1 * len(new))
            reward = novelty - cost * COST_SCALE - (0.05 if not new else 0.0)
            info.update({"new_items": len(new), "matched_items": len(matched)})

        elif action == "EXAM":
            items = state.case.objective
            matched = self._match_items(items, keywords)
            new = [index for index in matched if index not in state.disclosed_objective]
            state.disclosed_objective.update(new)
            cost = ACTION_COST["EXAM"] * max(1, len(keywords))
            body = "\n".join(f"- {items[index].content}" for index in matched) or "该项检查不适用或结果无异常。"
            observation = f"【检查结果】\n{body}"
            novelty = min(0.3, 0.1 * len(new))
            reward = novelty - cost * COST_SCALE - (0.05 if not new else 0.0)
            info.update({"new_items": len(new), "matched_items": len(matched)})

        else:  # KNOWLEDGE
            results = self.knowledge_base.search(keywords, top_k=3)
            body = self.knowledge_base.format_results(results)
            cost = ACTION_COST["KNOWLEDGE"]
            seen = any(query["keywords"] == keywords for query in state.knowledge_queries)
            state.knowledge_queries.append({"keywords": keywords, "hits": len(results)})
            observation = f"【知识库】\n{body}"
            reward = (0.05 if results and not seen else -0.05) - cost * COST_SCALE
            info.update({"hits": len(results), "repeated": seen})

        state.action_counts[action] += 1
        state.cost += cost

        reward = clip_step_reward(reward)
        state.total_reward += reward
        observation += f"\n\n（本轮成本 {cost} 元，累计 {state.cost:.0f} 元，剩余 {state.max_steps - state.turn} 轮）"
        return observation, reward

    def _handle_final(self, parsed: Dict[str, Any], info: Dict[str, Any]) -> Tuple[str, float]:
        state = self.state
        diagnosis = parsed.get("diagnosis", "").strip()
        treatment = parsed.get("treatment", "").strip()
        reward = self._finalize(reason="final", diagnosis=diagnosis, treatment=treatment)
        info.update({"breakdown": state.breakdown})
        return (
            f"【问诊结束】诊断：{diagnosis or '未给出'}\n处置：{treatment or '未给出'}\n"
            f"本题得分：{state.breakdown.get('score', 0):.2f}",
            reward,
        )

    def _finalize(self, reason: str, diagnosis: str, treatment: str) -> float:
        state = self.state
        state.done = True
        state.done_reason = reason
        state.diagnosis = diagnosis
        state.treatment = treatment

        if not diagnosis and not treatment:
            breakdown = {"diagnosis": 0.0, "treatment": 0.0, "safety": 0.0, "score": 0.0}
        else:
            breakdown = terminal_breakdown(diagnosis, treatment, {
                "diagnosis": state.case.ground_truth.diagnosis,
                "treatment": state.case.ground_truth.treatment,
                "avoid": state.case.ground_truth.avoid,
            })

        breakdown["efficiency"] = efficiency_reward(
            state.action_counts["ASK"], state.action_counts["EXAM"], state.action_counts["KNOWLEDGE"]
        )
        state.breakdown = breakdown

        terminal = breakdown["score"] - state.cost * COST_SCALE
        terminal = clip_step_reward(terminal)
        state.total_reward += terminal
        return terminal

    # -- 工具 -------------------------------------------------------------

    @staticmethod
    def _match_items(items: Sequence[Any], keywords: Sequence[str]) -> List[int]:
        """双向子串匹配，忽略长度小于 2 的词"""
        matched: List[int] = []
        for index, item in enumerate(items):
            hit = False
            for keyword in keywords:
                needle = keyword.lower().strip()
                if len(needle) < 2:
                    continue
                for item_keyword in item.keywords:
                    haystack = item_keyword.lower().strip()
                    if needle in haystack or haystack in needle:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                matched.append(index)
        return matched


def make_env(env_config: Optional[Dict[str, Any]] = None) -> MedAgentEnv:
    return MedAgentEnv(env_config=env_config)