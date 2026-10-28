#!/bin/bash
# ==============================================================================
# SmolVLA (0.45B) Fine-Tuning on RTX 4090 (24GB VRAM)
# Lightweight VLM (SmolVLM2-500M-Video) + Flow Matching Action Head
# ==============================================================================

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_ROOT=${DATASET_ROOT:-"datasets/wipes-to-green-plate-01"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/smolvla_wipes_ck30_bs64_30000"}

echo "=== [SmolVLA 0.45B Fine-Tuning on RTX 4090] ==="
echo "Dataset: $DATASET_ROOT"
echo "Output Directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=local/wipes-to-green-plate-01 \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=smolvla \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_wipes_30k \
  --policy.device=cuda \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --batch_size=64 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false
