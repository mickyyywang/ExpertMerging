#!/bin/bash
# Qwen2.5-7B Expert Merging 训练脚本
# 将 3 个 expert agent 模型合并到 Qwen2.5-7B-Instruct base model 上

set -e

MODEL_DIR="/primus_xpfs_workspace_T04/wmq/merge/models"
BASE_MODEL="${MODEL_DIR}/Qwen2.5-7B-Instruct"

EXPERT_1="${MODEL_DIR}/Qwen2.5-7B-Instruct-ToolRL-grpo-cold"   # Tool calling
EXPERT_2="${MODEL_DIR}/RL-MemoryAgent-7B"                       # Memory agent
EXPERT_3="${MODEL_DIR}/ReasonFlux-Coder-7B"                     # Code reasoning

DATA_DIR="/primus_xpfs_workspace_T04/wmq/merge/ExpertMerging/dataset"    # 训练数据：ToolCall.json / Memory.json / Code.json
OUTPUT_DIR="results/logs"

# ============================================================
# Expert Merging（可学习权重，推荐）
# ============================================================
cd "$(dirname "$0")"

python model_merging.py \
    --method expert_merging \
    --base_model "${BASE_MODEL}" \
    --expert_models "${EXPERT_1}" "${EXPERT_2}" "${EXPERT_3}" \
    --data_dir "${DATA_DIR}" \
    --output_path "${OUTPUT_DIR}" \
    --num_epochs 3 \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --mixed_precision bf16 \
    --sparsity_reg_weight 0.1 \
    --weight_coeffs_init_value 0.1 \
    --max_length 2048 \
    --samples_per_task 100 \
    --seed 42 \
    --exclude_params ".*lm_head.*" ".*norm.*" ".*embed_tokens.*" ".*bias.*"

# ============================================================
# 如果只想用 Task Arithmetic（不需要数据），取消下面的注释：
# ============================================================
# python model_merging.py \
#     --method task_arithmetic \
#     --base_model "${BASE_MODEL}" \
#     --expert_models "${EXPERT_1}" "${EXPERT_2}" "${EXPERT_3}" \
#     --scaling_coefficient 0.3 \
#     --output_path "${OUTPUT_DIR}" \
#     --exclude_params ".*lm_head.*" ".*norm.*" ".*embed_tokens.*" ".*bias.*"
