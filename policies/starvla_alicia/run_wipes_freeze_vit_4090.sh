#!/usr/bin/env bash
# ==============================================================================
# StarVLA (Qwen3.5-2B + QwenGR00T) 湿巾任务 — 冻结 ViT 视觉编码器训练脚本 (Single RTX 4090)
# ------------------------------------------------------------------------------
# 核心改进:
#   1. 冻结 Qwen3.5-VL 视觉塔 (freeze_modules: "qwen_vl_interface.model.model.visual")，
#      防止 150M 随机初始化 DiT 动作头在前几千步的高幅噪声梯度冲垮预训练视觉空间表征。
#   2. per_device_batch_size=1 + grad_accum=32 (等效 BS=32)，确保单卡 4090 显存安全余量 (~15.5 GB)。
#   3. 对齐官方 Franka 真机配置: optimizer.weight_decay=1e-8 (因 StarVLA 不拆分 1D Norm 参数的 weight_decay)。
#   4. 训练总步数设为 10,000 步 (约 13.0 Epochs，对齐最佳泛化区间)，每 2,000 步保存一次。
#
# 启动命令 (在 4090 服务器上):
#   nohup bash /home/yunyi/Desktop/vla/starVLA/examples/realRobots/Alicia_D_wipes/run_wipes_freeze_vit_4090.sh \
#     > /home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_freeze_vit.log 2>&1 &
# ==============================================================================

set -euo pipefail
cd /home/yunyi/Desktop/vla/starVLA

source ~/miniconda3/bin/activate starvla

export STARVLA_DISABLE_DEEPSPEED=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

config_yaml="examples/realRobots/Alicia_D_wipes/bar/starvla_wipes_qwen3_5_2b_gr00t_freeze_vit.yaml"
run_id="wipes_qwen3_5_2b_gr00t_freeze_vit"

output_dir="playground/Checkpoints/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "=== Launching StarVLA (Frozen ViT) Alicia-D Training on 1x RTX 4090 ==="
echo "Config: ${config_yaml}"
echo "Run ID: ${run_id}"
echo "Output: ${output_dir}"

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 32 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --run_root_dir playground/Checkpoints \
  --run_id "${run_id}"
