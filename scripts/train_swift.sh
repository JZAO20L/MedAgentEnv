#!/usr/bin/env bash
# MedAgent + ms-swift GRPO 训练启动脚本
#
# 前置：
#   pip install ms-swift
#   python scripts/build_swift_dataset.py --mode gym --output assets/mock/swift_gym.jsonl
#
# 用法：
#   bash scripts/train_swift.sh
#   MODEL=Qwen/Qwen3-4B MAX_TURNS=20 bash scripts/train_swift.sh
#   PROFILE=env-only bash scripts/train_swift.sh     # 只用环境总奖励
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B}
DATASET=${DATASET:-assets/mock/swift_gym.jsonl}
OUTPUT=${OUTPUT:-output/swift/medagent-grpo}
MAX_TURNS=${MAX_TURNS:-16}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}
LORA=${LORA:-true}
PROFILE=${PROFILE:-hybrid}

if [ ! -f "$DATASET" ]; then
  echo "数据集不存在：$DATASET"
  echo "请先运行：python scripts/build_swift_dataset.py --mode gym --output $DATASET"
  exit 1
fi

COMMON_ARGS=(
  --rlhf_type grpo
  --model "$MODEL"
  --dataset "$DATASET"
  --external_plugins scripts/swift_plugin.py
  --use_gym_env true
  --multi_turn_scheduler gym_scheduler
  --gym_env medagent
  --max_turns "$MAX_TURNS"
  --num_generations "$NUM_GENERATIONS"
  --per_device_train_batch_size "$BATCH_SIZE"
  --learning_rate "$LR"
  --enable_thinking false
  --use_vllm true
  --vllm_mode colocate
  --output_dir "$OUTPUT"
  --save_steps 50
  --logging_steps 1
)

if [ "$LORA" = "true" ]; then
  COMMON_ARGS+=(--tuner_type lora --lora_rank 16 --lora_alpha 32)
fi

if [ "$PROFILE" = "env-only" ]; then
  # 环境 total_reward 单路（GYMScheduler 自动追加）
  echo "== profile: env-only =="
  swift rlhf "${COMMON_ARGS[@]}" --reward_weights 1.0
else
  # 环境奖励 + 4 路规则奖励；reward_weights 末位对应 gym total_reward
  echo "== profile: hybrid =="
  swift rlhf "${COMMON_ARGS[@]}" \
    --reward_funcs med_diagnosis med_treatment med_safety med_efficiency med_cost \
    --reward_weights 0.4 0.3 0.2 0.2 0.1 1.0
fi