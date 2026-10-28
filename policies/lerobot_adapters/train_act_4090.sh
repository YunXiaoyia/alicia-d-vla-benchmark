#!/bin/bash
# ==============================================================================
# ACT (80M) Training from Scratch on RTX 4090 (24GB VRAM)
# ResNet-18 ImageNet Backbone + CVAE Encoder + Transformer Decoder
# ==============================================================================

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_ROOT=${DATASET_ROOT:-"datasets/wipes-to-green-plate-01"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/act_wipes_to_green_plate_run1"}

echo "=== [ACT 80M Training on RTX 4090] ==="
echo "Dataset: $DATASET_ROOT"
echo "Output Directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=local/wipes-to-green-plate-01 \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_wipes_50k \
  --policy.device=cuda \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=50000 \
  --batch_size=4 \
  --num_workers=8 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false
