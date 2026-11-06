#!/usr/bin/env bash
# ==============================================================================
# Alicia-D 湿巾任务 — 热启动对照队列 (单变量: 动作头是否冷启动)
# ------------------------------------------------------------------------------
# 两组完全相同的超参, 唯一差别是 --reinit_action_head:
#   A. warm_full : 载入 turbovla_libero.pth 全部 672 个张量 (action_queries 由 12 线性插值到 30)
#   B. warm_body : 载入除 action_head.* 以外的全部张量 (DINOv3/BERT/交互层/文本投影), 动作头冷启动
#
# 其余设置 (依据见 TURBOVLA_HANDOFF.md 审核结论):
#   - 论文 §4.1 明确写出的: LIBERO 热启动, 12.5k 步
#   - 论文 Table 8 (仅 LIBERO/RoboTwin, 非真机) 借用的: AdamW(0.9,0.95) wd=1e-10, lr=5e-5, 恒定 LR, EMA=0.999, 仅冻 BERT
#   - 论文未写、为我们自行选择的: batch=64, warmup=500, chunk=30, 其余
#
# 启动:
#   nohup bash /home/yunyi/Desktop/vla/scripts/train_turbovla_wipes_warm_ab.sh \
#     > /home/yunyi/Desktop/vla/TurboVLA/logs/warm_ab_queue.log 2>&1 &
# ==============================================================================

set -uo pipefail
cd /home/yunyi/Desktop/vla/TurboVLA

source ~/miniconda3/bin/activate lerobot

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/mplconfig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/yunyi/Desktop/vla/TurboVLA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

CKPT=/home/yunyi/Desktop/vla/TurboVLA/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
mkdir -p logs

run_one() {
  local tag="$1"; shift
  echo "=== [$(date '+%F %T')] START ${tag} ==="
  mkdir -p "outputs/turbovla_wipes_${tag}"
  python experiments/wipes/train_wipes.py \
    --dataset_dir /home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01 \
    --stats_path /home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json \
    --dinov3_path /home/yunyi/Desktop/vla/models/dinov3-vitb16-pretrain-lvd1689m \
    --bert_path /home/yunyi/Desktop/vla/models/bert-base-uncased \
    --pretrained_ckpt "${CKPT}" \
    --freeze_text_encoder \
    --checkpoint_dir "outputs/turbovla_wipes_${tag}" \
    --checkpoint_prefix "turbovla_wipes_${tag}_step" \
    --action_dim 7 --state_dim 8 --chunk_size 30 --num_state_tokens 2 \
    --expected_image_size 256 --position_embedding view --encode_views_separately \
    --padding_strategy key_padding_mask --residual_style normalized --attention_backend manual \
    --text_padding_length 21 --state_norm_mode zscore --precision fp32 \
    --batch_size 32 --grad_accum_steps 2 --max_steps 12500 --warmup_steps 500 \
    --lr_scheduler_type warmup_constant --min_lr_ratio 1.0 \
    --head_lr 5e-5 --dinov3_lr 5e-5 --weight_decay 1e-10 --max_grad_norm 1.0 \
    --ema_decay 0.999 --save_steps 2500 --log_freq 50 --num_workers 8 \
    --split_ratio 1.0 --seed 42 \
    --use_wandb --wandb_project turbovla-wipes --wandb_run_name "turbovla_wipes_${tag}_12k5" \
    "$@" 2>&1 | tee "logs/train_wipes_${tag}.log"
  echo "=== [$(date '+%F %T')] END ${tag} (exit=${PIPESTATUS[0]}) ==="
}

run_one warm_full
run_one warm_body --reinit_action_head
echo "=== ALL DONE $(date '+%F %T') ==="
