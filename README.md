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

## 当前实现（v0.6.1）

```
数据合成        src/medagent/data_pipeline.py    Generate -> Review -> Rewrite -> Judge
Agentic 环境    src/medagent/env.py              reset/step/close，逐轮 reward，可接 ms-swift
奖励            src/medagent/rewards.py          诊断 / 治疗 / 安全 / 效率 / cost（纯函数，可单测）
swift 接入      scripts/swift_plugin.py          Env + 5 个 ORM 注册，--use_gym_env true
演示            scripts/cli.py + webui.py        单病例 / 批量 / 水平扫描 / 投屏页 / 会话 API
```

### 医生水平梯度（v0.6.1 新增）

`demo_agent.py` 的三档策略让"水平"真正影响结论与开销：专家只开必要检查、覆盖全部处置要点；
主治加"考虑…可能"并多开一项非必要检查；实习生结论含糊、检查开全套。同一病例因此可复现地出现梯度：

| skill | 档位 | 平均总分 | 平均成本 | 诊断命中 | 治疗覆盖 |
|---|---|---|---|---|---|
| 1.00 | 专家 | 0.958 | 125 元 | 0.92 | 1.00 |
| 0.60 | 主治 | 0.833 | 142 元 | 0.92 | 0.58 |
| 0.20 | 实习生 | 0.200 | 175 元 | 0.00 | 0.00 |

这条性质由 `tests/test_demo_gradient.py` 锁定，改动奖励或剧本时不会静默失效。

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/LEADERBOARD.md` | 评测口径与可引用数字（两套评分不可混表） |
| `docs/DEFENSE.md` | 专业实践答辩口径：提纲逐条、5 分钟时间轴、Q&A 预案 |
| `v0_5.md` | 技术文档：模块设计、成本与评测方案、v0.4→v0.5 取舍 |
| `exp/README.md` | 评测脚本用法与预配置模型 |
| `results/README.md` | 历史评测结果字段说明 |
| `data/README.md` | 真实数据集接入格式（MIMIC-IV 等需自行授权） |

## 已知缺口

- `results/`、`exp/output/` 下的原始 `checkpoint_*.jsonl` / `model_stats_*.json` 被 `.gitignore` 排除，
  仓库只保留汇总表；复现 1,000 条评测需自备 API Key。
- Agentic RL 的训练链路（环境注册 + 奖励 + GRPO 脚本）已打通，但**尚未产出训练前后的对比曲线**。
- `assets/mock/` 只有 6 个演示病例，规模不足以做统计结论；结论数字以 `docs/LEADERBOARD.md` 为准。

## 快速开始

```bash
# 1. 离线跑一个病例（mock 数据，无需 API Key）
python scripts/cli.py run --case-id mock-mi-001 --skill 0.9

# 2. 批量跑全部 mock 病例
python scripts/cli.py batch --n 6

# 3. 医生水平扫描（输出可直接贴进幻灯片的对比表）
python scripts/cli.py sweep --skills 1.0,0.9,0.6,0.3,0.2 --detail

# 4. 启动 WebUI（答辩投屏用）
python scripts/cli.py serve --port 8000
# 打开 http://127.0.0.1:8000/

# 5. 构造 ms-swift 数据集并训练（需要先 pip install ms-swift）
python scripts/cli.py build-swift-data --mode gym
PROFILE=hybrid bash scripts/train_swift.sh

# 6. 测试
python -m pytest tests -q
```

**OpenEnv 风格接口**（与 `Env.reset/step` 同语义，可独立驱动）：

```bash
POST /api/reset  {"case_id": "mock-mi-001", "max_steps": 16}
POST /api/step   {"session_id": "...", "action": "<act><action>ASK</action>..."}
```

**数据说明**：仓库自带 `assets/mock/` 下 6 个跨科室病例、小知识库与价格清单，用于演示与测试；
MIMIC-IV、371k QA、真实病例库需自行授权后放入 `data/`，格式见 `data/README.md`。
