# MedAgent

## 设计核心:mocking env for benchmarking and agentic RL

目的:考察和训练LLM在模拟医院环境中,使用提供的医疗相关tool和自主查询资料协助疾病诊断/疾病治疗等任务.

传统相关研究存在的问题:大量使用LLM+角色扮演,或者以一些基本的workflow和SingleAgent作为基础组成部分搭建MultiAgent环境,API+MultiAgent系统带来天然的不确定性,高方差,高延时,不适于作为benchmark和作为agentic-RL训练环境

我们的实现方案:
将所有Agent可能需要使用的外部工具(这些工具集合起来用于代表医院环境)都定义为格式统一的tooluse交互函数;
通过合理的抽象,使用足量合成患者详细数据,
预设两类tooluse:
- 测量(获取)患者生理信息;通过访问数据实现
- 检索相关知识;通过预建立教材+百科+推理数据集的知识库+RAG实现;

---

## 当前实现（v0.6.0）

```
数据合成        src/medagent/data_pipeline.py    Generate -> Review -> Rewrite -> Judge
Agentic 环境    src/medagent/env.py              reset/step/close，逐轮 reward，可接 ms-swift
奖励            src/medagent/rewards.py          诊断 / 治疗 / 安全 / 效率 / cost
swift 接入      scripts/swift_plugin.py          Env + ORM 注册，--use_gym_env true
演示            scripts/cli.py + webui.py        单病例 / 批量 / 投屏页 / 会话 API
```

## 快速开始

```bash
# 1. 离线跑一个病例（mock 数据，无需 API Key）
python scripts/cli.py run --case-id mock-mi-001 --skill 0.9

# 2. 批量跑全部 mock 病例
python scripts/cli.py batch --n 6

# 3. 启动 WebUI（答辩投屏用）
python scripts/cli.py serve --port 8000
# 打开 http://127.0.0.1:8000/

# 4. 构造 ms-swift 数据集并训练（需要先 pip install ms-swift）
python scripts/cli.py build-swift-data --mode gym
PROFILE=hybrid bash scripts/train_swift.sh

# 5. 测试
python -m pytest tests -q
```

**OpenEnv 风格接口**（与 `Env.reset/step` 同语义，可独立驱动）：

```bash
POST /api/reset  {"case_id": "mock-mi-001", "max_steps": 16}
POST /api/step   {"session_id": "...", "action": "<act><action>ASK</action>..."}
```

**数据说明**：仓库自带 `assets/mock/` 下 6 个跨科室病例、小知识库与价格清单，用于演示与测试；
MIMIC-IV、371k QA、真实病例库需自行授权后放入 `data/`，格式见 `data/README.md`。
