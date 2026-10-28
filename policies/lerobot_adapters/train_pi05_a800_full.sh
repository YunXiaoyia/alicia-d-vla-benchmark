#!/bin/bash
# ==============================================================================
# Pi0.5 (4.14B) Full Fine-Tuning on A800 (80GB VRAM)
# Unfreezes both PaliGemma-3B Vision-Language Backbone and Action Expert (300M)
# ==============================================================================

set -e

# Environment Variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb

DATASET_ROOT=${DATASET_ROOT:-"datasets/wipes-to-green-plate-01"}
BASE_MODEL=${BASE_MODEL:-"models/pi05_base"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/pi05_wipes_a800_full"}

echo "=== [Pi0.5 4.14B Full Fine-Tuning on A800] ==="
echo "Dataset: $DATASET_ROOT"
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=local/wipes-to-green-plate-01 \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$BASE_MODEL" \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name=a800_full_30k_bs8 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --batch_size=8 \
  --num_workers=8 \
  --prefetch_factor=2 \
  --dataloader_multiprocessing_context=spawn \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable=true \
  --wandb.project=pi05_wipes \
  --policy.push_to_hub=false
