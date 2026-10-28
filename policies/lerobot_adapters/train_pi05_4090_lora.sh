#!/bin/bash
# ==============================================================================
# Pi0.5 (4.14B) Parameter-Efficient LoRA Fine-Tuning on RTX 4090 (24GB VRAM)
# Uses LoRA rank=16, alpha=32 on attention projections (q_proj, v_proj)
# ==============================================================================

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_ROOT=${DATASET_ROOT:-"datasets/wipes-to-green-plate-01"}
BASE_MODEL=${BASE_MODEL:-"models/pi05_base"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/pi05_wipes_4090_lora_ck30_bs3_30000"}

echo "=== [Pi0.5 4.14B PEFT LoRA Fine-Tuning on RTX 4090] ==="
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
  --job_name=pi05_wipes_4090_lora \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.use_peft=true \
  --policy.peft_type=lora \
  --policy.lora_r=16 \
  --policy.lora_alpha=32 \
  --policy.optimizer_lr=2.5e-4 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --batch_size=3 \
  --num_workers=6 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false
