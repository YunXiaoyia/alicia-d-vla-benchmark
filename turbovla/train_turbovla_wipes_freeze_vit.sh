#!/usr/bin/env bash
# ==============================================================================
# TurboVLA 湿巾抓取任务 — 冻结 DINOv3-ViT 视觉主干训练脚本 (Single RTX 4090 24GB)
# ------------------------------------------------------------------------------
# 核心改进 (对齐官方 TurboVLA + SmolVLA/Pi0.5 防过拟合机制):
#   1. 冻结 DINOv3 视觉主干 (--freeze_vision) + 保留原版真实世界 DINOv3 权重 (--keep_real_dinov3)
#      避免 DINOv3 在 71 条固定机位轨迹上死记背景像素，同时避免被 LIBERO 180° 翻转仿真域污染。
#   2. 预加载 LIBERO 40k / GroundingDINO 的跨模态交互层 (vision_language_interaction) 与动作头，
#      杜绝 6 层双向融合 Transformer 随机初始化导致的梯度崩塌。
#   3. 控制合理 Epoch 数: Global Batch Size = 64 (batch_size=32 * grad_accum=2, 1 Epoch ≈ 383 步)，
#      总步数 8,000 步 (≈ 20.8 Epochs)，每 2,000 步保存一次权重，采用官方 lr=5e-5 + weight_decay=1e-4。
#
# 启动命令 (在 4090 服务器上):
#   nohup bash /home/yunyi/Desktop/vla/scripts/train_turbovla_wipes_freeze_vit.sh \
#     > /home/yunyi/Desktop/vla/TurboVLA/logs/train_wipes_freeze_vit.log 2>&1 &
# ==============================================================================

set -euo pipefail
cd /home/yunyi/Desktop/vla/TurboVLA

source ~/miniconda3/bin/activate lerobot

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mplconfig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/yunyi/Desktop/vla/TurboVLA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs outputs/turbovla_wipes_freeze_vit

python experiments/wipes/train_wipes.py \
  --dataset_dir /home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01 \
  --stats_path /home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json \
  --dinov3_path /home/yunyi/Desktop/vla/models/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path /home/yunyi/Desktop/vla/models/bert-base-uncased \
  --pretrained_ckpt /home/yunyi/Desktop/vla/TurboVLA/outputs/libero_g128_40k/turbovla_step_40000.pth \
  --keep_real_dinov3 \
  --freeze_vision \
  --freeze_text_encoder \
  --checkpoint_dir outputs/turbovla_wipes_freeze_vit \
  --checkpoint_prefix turbovla_wipes_freeze_vit_step \
  --action_dim 7 \
  --chunk_size 30 \
  --state_dim 8 \
  --batch_size 32 \
  --grad_accum_steps 2 \
  --max_steps 8000 \
  --warmup_steps 500 \
  --head_lr 5e-5 \
  --dinov3_lr 0.0 \
  --weight_decay 1e-4 \
  --save_steps 2000 \
  --log_freq 50 \
  --num_workers 8 \
  --split_ratio 1.0 \
  --seed 42 \
  --use_wandb \
  --wandb_project turbovla-wipes \
  --wandb_run_name turbovla_wipes_freeze_vit_bs64_8k
