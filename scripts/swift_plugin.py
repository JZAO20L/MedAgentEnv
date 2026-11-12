"""ms-swift 外部插件：注册 MedAgent 环境与奖励函数

训练时这样加载：

    swift rlhf ... \
        --external_plugins scripts/swift_plugin.py \
        --use_gym_env true --multi_turn_scheduler gym_scheduler --gym_env medagent

也可以只注册奖励函数：

    --reward_funcs med_diagnosis med_treatment med_safety med_efficiency med_cost
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # 训练环境才有 swift
    from swift.rollout.gym_env import Env, envs
    SWIFT_GYM_AVAILABLE = True
except Exception:  # noqa: BLE001
    class Env:  # type: ignore[no-redef]
        """swift 未安装时的占位，便于本地 import 检查"""

        def __init__(self, env_config=None):
            self.env_config = env_config or {}

    envs = {}
    SWIFT_GYM_AVAILABLE = False

from src.medagent.env import MedAgentEnv
from src.medagent.swift_rewards import register as register_rewards


class MedAgentSwiftEnv(Env):
    """把 MedAgentEnv 适配成 swift 的 Env 接口"""

    def __init__(self, env_config=None):
        try:
            super().__init__(env_config or {})
        except TypeError:
            super().__init__()
            self.env_config = env_config or {}
        self._env = MedAgentEnv(env_config=env_config or {})

    async def reset(self, config):
        return await self._env.reset(config)

    async def step(self, action):
        return await self._env.step(action)

    async def close(self):
        return await self._env.close()


envs["medagent"] = MedAgentSwiftEnv
register_rewards()

__all__ = ["MedAgentSwiftEnv", "SWIFT_GYM_AVAILABLE", "envs"]