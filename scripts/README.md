# MedAgent - 脚本目录

## 概述

本目录存放 MedAgent 项目的所有可执行脚本，包括训练、评测、部署和环境配置。

## 脚本列表

### Agentic RL 训练

| 脚本 | 说明 |
|------|------|
| `agentic_rl.py` | Agentic RL 训练主脚本 |
| `agentic_rl.sh` | Agentic RL 实验运行脚本 (4 个实验) |

### 模型评测

| 脚本 | 说明 |
|------|------|
| `eval.py` | 单模型评测脚本 |
| `benchmark.py` | 多模型基准评测脚本 |
| `test_models.py` | 模型测试脚本 |

### 模型部署与测试

| 脚本 | 说明 |
|------|------|
| `model_test.sh` | 本地模型部署与测试脚本 (vLLM + LoRA) |
| `rag.py` | 独立 RAG CLI 工具 |

### 演示与 Swift 接入

| 脚本 | 说明 |
|------|------|
| `cli.py` | 命令行入口（run / batch / serve / build-swift-data） |
| `swift_plugin.py` | ms-swift 外部插件：注册 MedAgent 环境与 5 路奖励 |
| `build_swift_dataset.py` | 把病例 JSONL 转成 swift 数据集（gym / prompt 两种模式） |
| `train_swift.sh` | `swift rlhf` GRPO 启动脚本（env-only / hybrid 两套权重） |

### 数据处理

| 脚本 | 说明 |
|------|------|
| `data_pipeline.py` | 数据合成管线 (fast / quality) |
| `split_dataset.py` | 数据集划分脚本 |

### 辅助脚本

| 脚本 | 说明 |
|------|------|
| `setup_conda_env.sh` | Conda 环境配置脚本 |
| `train_multi_gpu.sh` | 多 GPU 训练脚本 |
| `train_multi_gpu.bat` | 多 GPU 训练脚本 (Windows) |
| `build_knowledge_base.sh` | 旧知识库构建脚本 (已废弃) |
| `batch_eval.sh` | 批量评测脚本 (已废弃) |
| `deprecated/` | 已废弃脚本集合 |

## 详细用法

### 1. Agentic RL 训练 (`agentic_rl.py` / `agentic_rl.sh`)

**功能**: 使用 GRPO 算法训练医疗诊断 Agent。

#### 1.1 一键运行 4 个实验

```bash
bash scripts/agentic_rl.sh
```

**实验列表**:

| 实验 | 医疗奖励 | 效率奖励 | Cost 奖励 | 输出目录 |
|------|---------|---------|----------|---------|
| medical_only | ✅ | ❌ | ❌ | `output/agentic_rl/medical_only` |
| medical_efficiency | ✅ | ✅ | ❌ | `output/agentic_rl/medical_efficiency` |
| medical_cost | ✅ | ❌ | ✅ | `output/agentic_rl/medical_cost` |
| medical_efficiency_cost | ✅ | ✅ | ✅ | `output/agentic_rl/medical_efficiency_cost` |

#### 1.2 单独运行

```bash
python scripts/agentic_rl.py \
    --model /root/autodl-tmp/models/Qwen/Qwen3-4B \
    --data data/datasets/train.jsonl \
    --output-dir output/agentic_rl/medical_only \
    --max-steps 500 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --use-lora \
    --use-vllm \
    --disable-tool-efficiency

# 或者启用所有奖励
python scripts/agentic_rl.py \
    --model /root/autodl-tmp/models/Qwen/Qwen3-4B \
    --data data/datasets/train.jsonl \
    --output-dir output/agentic_rl/medical_efficiency_cost \
    --max-steps 500 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --use-lora \
    --use-vllm \
    --enable-cost-reward
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | Qwen/Qwen3-4B | 基座模型路径 |
| `--data` | data/datasets/train.jsonl | 训练数据路径 |
| `--output-dir` | output/agentic_rl | 输出目录 |
| `--max-steps` | 500 | 最大训练步数 |
| `--batch-size` | 8 | 批次大小 |
| `--learning-rate` | 1e-5 | 学习率 |
| `--use-lora` | False | 使用 LoRA 微调 |
| `--use-vllm` | False | 使用 vLLM 加速 |
| `--disable-tool-efficiency` | False | 禁用工具效率奖励 |
| `--enable-cost-reward` | False | 启用 Cost 奖励 |
| `--disable-judger` | False | 禁用 Judger 评分 |
| `--rule-only-judger` | True | Judger 仅使用规则匹配 (无需 API) |
| `--kb-path` | data/knowledge_dataset/ResponseMed.json | 知识库路径 |

### 2. Benchmark 评测 (`benchmark.py`)

**功能**: 对多个模型在基准数据上进行全量评测。

```bash
# 评测所有预配置模型
python exp/benchmark.py --all --n 1000 --max-workers 5

# 评测指定模型
python exp/benchmark.py --models gpt-5.4,qwen3.5-plus --n 1000

# 使用本地部署模型
python exp/benchmark.py --models qwen3-4b-base \
    --base-url http://localhost:8000/v1 \
    --n 100
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--models` | 前 5 个模型 | 评测模型列表 |
| `--all` | False | 评测所有预配置模型 |
| `--n` | 1000 | 评测数据量 |
| `--max-workers` | 5 | 并发线程数 |
| `--output` | exp/output | 输出目录 |
| `--base-url` | None | 自定义模型 base URL |
| `--api-key` | None | 自定义 API key |

### 3. 模型部署测试 (`model_test.sh`)

**功能**: 部署本地 vLLM 模型并进行评测。

```bash
# 测试 base 模型
bash scripts/model_test.sh --model-path Qwen/Qwen3-4B --n 100

# 测试 LoRA 微调模型
bash scripts/model_test.sh \
    --model-path Qwen/Qwen3-4B \
    --lora-path output/agentic_rl/medical_only \
    --n 100

# 自定义模型名称
bash scripts/model_test.sh \
    --model-path Qwen/Qwen3-4B \
    --model-name qwen3-4b-base \
    --n 500
```

### 4. 环境配置 (`setup_conda_env.sh`)

**功能**: 创建 medagent conda 环境，安装所有依赖。

```bash
bash scripts/setup_conda_env.sh
```

**环境配置**:
- Python 3.11
- PyTorch (CUDA 12.1)
- TRL >= 0.27.0
- Transformers 5.0+ (开发版)
- vLLM
- PEFT, Accelerate, Datasets

### 5. 数据合成管线 (`data_pipeline.py`)

**功能**: 从种子病例生成可用于评测/训练的 `MedicalCase`。

```bash
# 质量模式（Generate -> Review -> Rewrite -> Judge，4 次调用/样本）
python scripts/data_pipeline.py \
    --seed-data data/seed_cases \
    --output output/synthesis/cases.jsonl

# 快速模式（Generate -> Judge，2 次调用/样本）
python scripts/data_pipeline.py --mode fast --max-samples 200

# 三个裁判投票（3/3 通过）
python scripts/data_pipeline.py \
    --judge qwen3.5-plus@https://coding.dashscope.aliyuncs.com/v1@DASHSCOPE_API_KEY_CP \
    --judge glm-5@https://open.bigmodel.cn/api/paas/v4/@GLM_API_KEY \
    --judge baichuan-m2@http://localhost:8200/v1@BAICHUAN_API_KEY

# 无 API Key 时验证管线
python scripts/data_pipeline.py --dry-run --max-samples 20
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--seed-data` | data/seed_cases | 种子文件或按科室分文件的目录 |
| `--output` | output/synthesis/cases.jsonl | 通过样本（扁平 MedicalCase JSONL） |
| `--checkpoint` | `<output>.ckpt.jsonl` | 断点文件，默认按 seed_id 续跑 |
| `--mode` | quality | `fast` 2 次调用 / `quality` 4 次调用 |
| `--start-idx` / `--end-idx` | 0 / None | 种子切片 |
| `--shard-size` | 500 | 每片样本数，按科室轮询保证分布均衡 |
| `--shard-interval` | 0 | 片间暂停秒数，用于规避限流 |
| `--workers` | 4 | 并发样本数 |
| `--generator` / `--reviewer` / `--rewriter` | 见文档 | `model@base_url[@api_key_env]` |
| `--judge` | baichuan-m2@localhost:8200 | 可重复；默认全部裁判通过才算通过 |
| `--min-passing` | 裁判数 | 最低通过票数 |
| `--dry-run` | False | 离线模式，不调用 API，仅验证管线与校验 |

**入库校验**（`validate_case`）：主诉长度、主观/客观条数、keywords 非空且不重复、content 长度、
diagnosis/treatment 非空、主诉不得泄露诊断名。

**说明**: 设计文档中的 `data_pipeline_v2.py`（分片 + 2 小时间隔）、`data_pipeline_oneshot.py`、
`mimic_pipeline.py` 目前未包含在本仓库；对应的分片、断点与间隔能力已并入 `data_pipeline.py`。

### 6. RAG CLI 工具 (`rag.py`)

**功能**: 独立的 RAG 检索测试工具（使用旧 embedding+reranker 方案）。

```bash
# 构建知识库
python scripts/rag.py \
    --build \
    --data data/knowledge_dataset/ResponseMed.json \
    --db data/knowledge_db

# 检索
python scripts/rag.py \
    --search "高血压的治疗方案" \
    --db data/knowledge_db \
    --top-k 10 \
    --rerank 3
```

### 7. 多 GPU 训练 (`train_multi_gpu.sh`)

**功能**: 使用 torchrun 进行多 GPU 分布式训练。

```bash
bash scripts/train_multi_gpu.sh \
    --model Qwen/Qwen3-4B \
    --data data/datasets/train.jsonl \
    --output output/trained_model \
    --nproc-per-node 4 \
    --max-steps 500
```

### 8. 演示 CLI (`cli.py`)

```bash
python scripts/cli.py version
python scripts/cli.py cases
python scripts/cli.py run --case-id mock-mi-001 --skill 0.9
python scripts/cli.py batch --n 6
python scripts/cli.py serve --port 8000      # WebUI + 会话 API
python scripts/cli.py build-swift-data --mode gym
```

`serve` 同时提供 OpenEnv 风格接口：

| 接口 | 说明 |
|------|------|
| `GET /api/cases` | 病例列表 |
| `POST /api/run` | 跑完一个 episode（演示用） |
| `POST /api/reset` | 创建会话，返回 observation / system_message |
| `POST /api/step` | 在会话上执行一个动作，返回 reward / done / info |

### 9. ms-swift GRPO 训练 (`swift_plugin.py` / `train_swift.sh`)

```bash
# 1) 构造 gym 模式数据集（每行一个 env_config）
python scripts/build_swift_dataset.py --mode gym --output assets/mock/swift_gym.jsonl

# 2) 训练
bash scripts/train_swift.sh
# 只使用环境总奖励
PROFILE=env-only bash scripts/train_swift.sh
```

插件注册内容：

- `envs["medagent"] = MedAgentSwiftEnv`：包装 `src/medagent/env.py`
- `orms["med_diagnosis" | "med_treatment" | "med_safety" | "med_efficiency" | "med_cost"]`

默认 hybrid 权重：`0.4 / 0.3 / 0.2 / 0.2 / 0.1 + 1.0`（末位为 gym `total_reward`）。

## 环境变量配置

运行脚本前需要设置以下环境变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `DASHSCOPE_API_KEY_CP` | 百炼 codingplan API key | `sk-xxx` |
| `DASHSCOPE_API_KEY` | 百炼 normal API key | `sk-xxx` |
| `BAICHUAN_API_KEY` | 百川 API key | `sk-xxx` |
| `302_API_KEY` | 302AI API key | `sk-xxx` |
| `CUSTOM_API_KEY` | 自定义 API key | `sk-xxx` |

```bash
# 方式 1: 直接设置
export BAICHUAN_API_KEY="sk-xxx"

# 方式 2: 使用 .env 文件
# 在项目根目录创建 .env 文件
BAICHUAN_API_KEY=sk-xxx
DASHSCOPE_API_KEY_CP=sk-xxx
302_API_KEY=sk-xxx
```

## 相关文档

- `../README.md`: 项目概述
- `../src/README.md`: 源代码目录说明
- `../exp/README.md`: 实验目录说明
- `../data/README.md`: 数据目录说明
