"""
MedAgent - 医疗智能 Agent 核心模块
"""

from .llm import LLMClient
from .version import __version__
# from .embedding import EmbeddingClient
# from .reranker import RerankerClient
# from .knowledge_base import KnowledgeBase
from .knowledge_tool_v2 import KeywordKnowledgeBase, KeywordKnowledgeTool
from .tool import BaseTool, AskTool, ExamTool, KnowledgeTool, ToolManager
from .agent import MedAgent, AgentState
from .judger import Judger, EvalResult, EfficiencyStats
from .cost import CostEvaluator, CostItem, CostResult, CostEvaluator as CostEstimator
from .data_pipeline import (
    CaseSynthesizer,
    PipelineConfig,
    RoleConfig,
    SeedRecord,
    load_seeds,
    make_shards,
    run_pipeline,
    validate_case,
)
from .env import CaseCatalog, MedAgentEnv, MockKnowledgeBase, parse_action
from .demo_agent import run_episode, scripted_action
from .rewards import (
    cost_penalty,
    diagnosis_reward,
    efficiency_reward,
    safety_reward,
    terminal_breakdown,
    treatment_reward,
)

__all__ = [
    "__version__",
    "LLMClient",
    # "EmbeddingClient",
    # "RerankerClient",
    # "KnowledgeBase",
    "KeywordKnowledgeBase",
    "KeywordKnowledgeTool",
    "BaseTool",
    "AskTool",
    "ExamTool",
    "KnowledgeTool",
    "ToolManager",
    "MedAgent",
    "AgentState",
    "Judger",
    "EvalResult",
    "EfficiencyStats",
    "CostEvaluator",
    "CostEstimator",
    "CostItem",
    "CostResult",
    "CaseSynthesizer",
    "PipelineConfig",
    "RoleConfig",
    "SeedRecord",
    "load_seeds",
    "make_shards",
    "run_pipeline",
    "validate_case",
    "CaseCatalog",
    "MedAgentEnv",
    "MockKnowledgeBase",
    "parse_action",
    "run_episode",
    "scripted_action",
    "cost_penalty",
    "diagnosis_reward",
    "efficiency_reward",
    "safety_reward",
    "terminal_breakdown",
    "treatment_reward",
]
