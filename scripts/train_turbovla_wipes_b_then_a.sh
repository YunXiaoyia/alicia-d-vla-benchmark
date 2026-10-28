#!/bin/bash
# ==============================================================================
# TurboVLA 湿巾任务对比消融流水线 (等效 Global Batch Size = 128)
# 顺序: 先跑实验 B (从头训练) -> 跑完自动跑实验 A (LIBERO热启动)
# 用法:
#   nohup bash /home/yunyi/Desktop/vla/scripts/train_turbovla_wipes_b_then_a.sh > /home/yunyi/Desktop/vla/TurboVLA/logs/pipeline_b_then_a.log 2>&1 &
# ==============================================================================

set -e
cd /home/yunyi/Desktop/vla/TurboVLA

source ~/miniconda3/bin/activate lerobot

export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/mplconfig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/yunyi/Desktop/vla/TurboVLA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs outputs/turbovla_wipes_b_scratch outputs/turbovla_wipes_a_warmstart

echo "=============================================================================="
echo ">>> [消融实验流水线启动 (Global Batch Size = 128)]"
echo ">>> 顺序: 实验 B (从头训练) -> 实验 A (LIBERO热启动)"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 阶段 1：实验 B —— 纯从头训练（Cold Start, 不加载 LIBERO 权重）
# ------------------------------------------------------------------------------
echo ""
echo "##############################################################################"
echo ">>> [1/2] 正在启动 实验 B: 纯从头初始化训练 (From Scratch)"
echo ">>> 日志文件: /home/yunyi/Desktop/vla/TurboVLA/logs/train_wipes_b_scratch.log"
echo ">>> 保存目录: /home/yunyi/Desktop/vla/TurboVLA/outputs/turbovla_wipes_b_scratch"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "##############################################################################"

python experiments/wipes/train_wipes.py \
  --dataset_dir /home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01 \
  --stats_path /home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json \
  --dinov3_path /home/yunyi/Desktop/vla/models/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path /home/yunyi/Desktop/vla/models/bert-base-uncased \
  --pretrained_ckpt "" \
  --checkpoint_dir outputs/turbovla_wipes_b_scratch \
  --checkpoint_prefix turbovla_wipes_scratch_step \
  --action_dim 7 \
  --chunk_size 30 \
  --state_dim 8 \
  --batch_size 64 \
  --grad_accum_steps 2 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --head_lr 1.5e-4 \
  --dinov3_lr 3.5e-5 \
  --save_steps 4000 \
  --log_freq 200 \
  --num_workers 8 \
  --split_ratio 1.0 \
  --seed 42 \
  --use_wandb \
  --wandb_project turbovla-wipes \
  --wandb_run_name turbovla_wipes_b_scratch \
  2>&1 | tee logs/train_wipes_b_scratch.log

echo ""
echo ">>> [1/2 完成] 实验 B (从头训练) 执行完毕！结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ------------------------------------------------------------------------------
# 阶段 2：实验 A —— LIBERO 40k 权重热启动微调（Warm Start）
# ------------------------------------------------------------------------------
echo "##############################################################################"
echo ">>> [2/2] 正在启动 实验 A: 加载 LIBERO 预训练权重热启动微调 (Warm Start)"
echo ">>> 预训练权重: /home/yunyi/Desktop/vla/TurboVLA/outputs/libero_g128_40k/turbovla_step_40000.pth"
echo ">>> 日志文件: /home/yunyi/Desktop/vla/TurboVLA/logs/train_wipes_a_warmstart.log"
echo ">>> 保存目录: /home/yunyi/Desktop/vla/TurboVLA/outputs/turbovla_wipes_a_warmstart"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "##############################################################################"

python experiments/wipes/train_wipes.py \
  --dataset_dir /home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01 \
  --stats_path /home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json \
  --dinov3_path /home/yunyi/Desktop/vla/models/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path /home/yunyi/Desktop/vla/models/bert-base-uncased \
  --pretrained_ckpt /home/yunyi/Desktop/vla/TurboVLA/outputs/libero_g128_40k/turbovla_step_40000.pth \
  --checkpoint_dir outputs/turbovla_wipes_a_warmstart \
  --checkpoint_prefix turbovla_wipes_warmstart_step \
  --action_dim 7 \
  --chunk_size 30 \
  --state_dim 8 \
  --batch_size 64 \
  --grad_accum_steps 2 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --head_lr 1.5e-4 \
  --dinov3_lr 3.5e-5 \
  --save_steps 4000 \
  --log_freq 200 \
  --num_workers 8 \
  --split_ratio 1.0 \
  --seed 42 \
  --use_wandb \
  --wandb_project turbovla-wipes \
  --wandb_run_name turbovla_wipes_a_warmstart \
  2>&1 | tee logs/train_wipes_a_warmstart.log

echo ""
echo "=============================================================================="
echo ">>> [消融实验流水线全部完成!] B(从头训练) 与 A(热启动) 均已跑完！"
echo ">>> 全部结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
