#!/usr/bin/env bash
# ==============================================================================
# StarVLA (Qwen3.5-2B + QwenGR00T) 湿巾任务 — 冻结整个 Qwen-VL (FixQwen) 训练脚本 (Single RTX 4090)
# ------------------------------------------------------------------------------
# 核心改进:
#   1. 冻结整个 Qwen3.5-2B 多模态主干 (freeze_modules: "qwen_vl_interface"，含 ViT 337M + LLM 1,793M)，
#      只训练 356.8M 的 DiT-B 动作头 (action_model)，100% 杜绝 VLM 在 81 条单任务数据上过拟合。
#   2. 无需存 LLM 梯度与优化器状态，开启 per_device_batch_size=4 + grad_accum=8 (等效 BS=32)，训练提速约 3 倍。
#   3. 对齐官方 Franka 真机配置: optimizer.weight_decay=1e-8，lr(action_model)=1e-4。
#   4. 训练总步数设为 10,000 步 (约 13.0 Epochs)，每 2,000 步保存一次。
#
# 启动命令 (在 4090 服务器上):
#   nohup bash /home/yunyi/Desktop/vla/starVLA/examples/realRobots/Alicia_D_wipes/run_wipes_fixqwen_4090.sh \
#     > /home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_fixqwen.log 2>&1 &
# ==============================================================================

set -euo pipefail
cd /home/yunyi/Desktop/vla/starVLA

source ~/miniconda3/bin/activate starvla

export STARVLA_DISABLE_DEEPSPEED=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

config_yaml="examples/realRobots/Alicia_D_wipes/bar/starvla_wipes_qwen3_5_2b_gr00t_fixqwen.yaml"
run_id="wipes_qwen3_5_2b_gr00t_fixqwen"

output_dir="playground/Checkpoints/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "=== Launching StarVLA (FixQwen / Full VLM Frozen) Alicia-D Training on 1x RTX 4090 ==="
echo "Config: ${config_yaml}"
echo "Run ID: ${run_id}"
echo "Output: ${output_dir}"

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --run_root_dir playground/Checkpoints \
  --run_id "${run_id}"
