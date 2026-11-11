#!/usr/bin/env bash
# ==============================================================================
# [A 组] TurboVLA (LIBERO 全量热启动 12.5k 步: warm_full) — 无平滑基线
# 硬件端口：机械臂 /dev/ttyACM0 | 全局相机 top=/dev/video8 | 腕部相机 wrist=/dev/video6
# ==============================================================================
set -euo pipefail

VLA_ROOT="/media/yunyi/Data/Desktop/work/vla"
export PYTHONPATH="${VLA_ROOT}/lerobot/src:${VLA_ROOT}/Alicia-D-SDK:${VLA_ROOT}/TurboVLA:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_PATH="${CKPT_PATH:-${VLA_ROOT}/weights/turbovla_wipes/turbovla_wipes_warm_full_step_12500.pth}"

/home/yunyi/miniconda3/envs/lerobot/bin/python \
  "${VLA_ROOT}/alicia-d-vla-benchmark/scripts/eval_alicia_turbovla.py" \
  --robot.type=alicia_d_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=alicia_duo_follower_arm \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30},
    "top":   {"type": "opencv", "index_or_path": "/dev/video8", "width": 640, "height": 480, "fps": 30}
  }' \
  --ckpt_path="${CKPT_PATH}" \
  --stats_path="${VLA_ROOT}/weights/turbovla_wipes/wipes_stats.json" \
  --dinov3_path="${VLA_ROOT}/models/dinov3-vitb16-pretrain-lvd1689m" \
  --bert_path="${VLA_ROOT}/models/bert-base-uncased" \
  --device=cuda \
  --use_ema=true \
  --chunk_size=30 \
  --n_action_steps=30 \
  --temporal_ensemble_coeff=0 \
  --task="Pick up the pack of cleaning wipes and place it on the green plate" \
  --duration=25 \
  --fps=30 \
  --num_episodes=1 \
  --record_eval=true \
  --eval_dataset_root="${VLA_ROOT}/eval_runs" \
  --eval_dataset_repo_id="local/eval_turbovla_warm_full_no_smooth" \
  --display_data=false \
  "$@"
