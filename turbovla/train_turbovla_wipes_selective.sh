#!/usr/bin/env bash
# ==============================================================================
# TurboVLA 湿巾任务 — 选择性热启动训练脚本（替代 warm_full / warm_body）
# ------------------------------------------------------------------------------
# 依据：docs/turbovla_realrobot_diagnosis.md
#
# 结论摘要（来自 13 组真机录包的实测）：
#   freeze_vit（视觉冻结 + 动作头载入 LIBERO）→ 真机 100% 成功
#   warm_body（视觉冻结 + 动作头冷启动）      → 抓到但无法撤离，超时卡在盘沿
#   warm_full（视觉解冻 + 5e-5 恒定 LR 洗 32.6 epoch）→ 离线 MAE 最好(0.754°)，真机最差
#
# 因此本脚本只提供两个 Stage，都 **完整载入 LIBERO 动作头**（绝不用 --reinit_action_head）：
#   A（推荐，先跑这个）: 复刻 freeze_vit —— 冻视觉、冻文本、只训融合层+动作头
#   B（仅当 A 明显欠拟合时才用）: 视觉解冻但 lr 降到 5e-6 + cosine 衰减到 0.1，短步数
#
# ⚠️ 复现陷阱（务必知晓）：
#   freeze_vit 的 checkpoint 里记录 min_lr_ratio=0.025、ema_decay=0.999，
#   但当前 trainer.py / train_wipes.py 的 --min_lr_ratio 默认值已被改为 1.0（恒定 LR）。
#   如果直接重跑旧的 train_turbovla_wipes_freeze_vit.sh，会 **静默训练成恒定 LR**，
#   与当初 100% 成功的那次并不等价。本脚本把所有超参显式钉死，不依赖任何默认值。
#
# 启动（在 4090 服务器上）：
#   nohup bash /home/yunyi/Desktop/vla/scripts/train_turbovla_wipes_selective.sh A \
#     > /home/yunyi/Desktop/vla/TurboVLA/logs/selective_A.log 2>&1 &
#   nohup bash /home/yunyi/Desktop/vla/scripts/train_turbovla_wipes_selective.sh B \
#     > /home/yunyi/Desktop/vla/TurboVLA/logs/selective_B.log 2>&1 &
#
# 可选环境变量：
#   VAL_SPLIT=0.9   留出 10% episode 做验证集（71 → 63 训练 / 8 验证）。
#                   **强烈建议开启**：现在 split_ratio=1.0 意味着你在用训练集 MAE 选模型，
#                   而 warm_full 已经证明这个指标与真机成功率负相关。
#   SEED=42
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

STAGE="${1:-A}"
VAL_SPLIT="${VAL_SPLIT:-1.0}"
SEED="${SEED:-42}"

DATA=/home/yunyi/Desktop/vla/datasets/wipes-to-green-plate-01
STATS=/home/yunyi/Desktop/vla/TurboVLA/experiments/wipes/wipes_stats.json
DINOV3=/home/yunyi/Desktop/vla/models/dinov3-vitb16-pretrain-lvd1689m
BERT=/home/yunyi/Desktop/vla/models/bert-base-uncased
LIBERO=/home/yunyi/Desktop/vla/TurboVLA/outputs/libero_g128_40k/turbovla_step_40000.pth
[ -f "$LIBERO" ] || LIBERO=/home/yunyi/Desktop/vla/TurboVLA/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth
[ -f "$LIBERO" ] || { echo "FATAL: LIBERO 预训练权重不存在，请检查路径"; exit 1; }
echo "使用仿真预训练权重: ${LIBERO}"

# ---- 所有 stage 共用的、必须显式钉死的参数（不要依赖默认值）----
COMMON=(
  --dataset_dir "$DATA"
  --stats_path "$STATS"
  --dinov3_path "$DINOV3"
  --bert_path "$BERT"
  --pretrained_ckpt "$LIBERO"
  --action_dim 7 --state_dim 8 --chunk_size 30 --num_state_tokens 2
  --expected_image_size 256 --text_padding_length 21
  --position_embedding view --encode_views_separately
  --padding_strategy key_padding_mask --residual_style normalized
  --attention_backend manual --interaction_precision fp32
  --state_norm_mode zscore --precision fp32
  --freeze_text_encoder
  --batch_size 32 --grad_accum_steps 2
  --max_grad_norm 1.0 --ema_decay 0.999
  --num_workers 8 --log_freq 50 --seed "$SEED"
  --split_ratio "$VAL_SPLIT"
)

case "$STAGE" in
  A)
    # ---------------------------------------------------------------------
    # Stage A：复刻 freeze_vit（真机 100% 成功的那次）
    #   - 视觉冻结，且保留 web 预训练的 DINOv3（不要把主干换成 LIBERO 仿真权重）
    #   - 完整载入 LIBERO 的融合层 + 动作头（7 维关节空间同维，可直接继承）
    #   - 只训融合层 + 动作头，8000 步 ≈ 20.9 epoch
    #   - wd 1e-4（真正则），LR 余弦衰减到 2.5%
    # ---------------------------------------------------------------------
    TAG=turbovla_wipes_selective_A
    OUT="outputs/${TAG}"
    mkdir -p "$OUT"
    python experiments/wipes/train_wipes.py \
      "${COMMON[@]}" \
      --keep_real_dinov3 --freeze_vision \
      --checkpoint_dir "$OUT" \
      --checkpoint_prefix "${TAG}_step" \
      --max_steps 8000 --warmup_steps 500 \
      --lr_scheduler_type warmup_cosine --min_lr_ratio 0.025 \
      --head_lr 5e-5 --dinov3_lr 0.0 \
      --weight_decay 1e-4 --head_weight_decay 1e-4 --dinov3_weight_decay 1e-4 \
      --save_steps 2000 \
      --use_wandb --wandb_project turbovla-wipes --wandb_run_name "${TAG}_8k"
    ;;

  B)
    # ---------------------------------------------------------------------
    # Stage B：受控解冻（**仅当 Stage A 真机明显欠拟合/泛化不足时才用**）
    #   - 解冻视觉，但 lr 只有 5e-6（Stage A 的 head lr 的 1/20），且 wd 1e-4
    #   - cosine 衰减到 0.1，4000 步（≈10.4 epoch），早停，不给过拟合留窗口
    #   - head lr 1e-4 补偿视觉端的保守
    #   仍然完整载入 LIBERO 动作头。
    # ---------------------------------------------------------------------
    TAG=turbovla_wipes_selective_B
    OUT="outputs/${TAG}"
    mkdir -p "$OUT"
    # 注意：这里**不传** --freeze_vision，视觉端以 5e-6 极小 lr 参与训练
    python experiments/wipes/train_wipes.py \
      "${COMMON[@]}" \
      --checkpoint_dir "$OUT" \
      --checkpoint_prefix "${TAG}_step" \
      --max_steps 4000 --warmup_steps 250 \
      --lr_scheduler_type warmup_cosine --min_lr_ratio 0.1 \
      --head_lr 1e-4 --dinov3_lr 5e-6 \
      --weight_decay 1e-4 --head_weight_decay 1e-4 --dinov3_weight_decay 1e-4 \
      --save_steps 1000 \
      --use_wandb --wandb_project turbovla-wipes --wandb_run_name "${TAG}_4k"
    ;;

  *)
    echo "用法: bash $0 {A|B}"
    exit 1
    ;;
esac

echo "=== [$(date '+%F %T')] STAGE ${STAGE} 完成 ==="
echo "权重目录: ${OUT}"
echo "下一步：拉到本地后用 eval_turbovla_smooth.sh 评测（记得先把 eval_runs 里同名目录改名备份）"
