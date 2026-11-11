#!/usr/bin/env bash
# ==============================================================================
# [实验 6/10] Pi0.5 (3B, A800 Full-FT 20k) — 开启 LeRobot 原生 RTC 实时流匹配引导平滑 (With RTC + Horizon=15)
# 硬件端口：机械臂 /dev/ttyACM0 | 全局相机 top=/dev/video8 | 腕部相机 wrist=/dev/video6
# ==============================================================================
set -euo pipefail

VLA_ROOT="/media/yunyi/Data/Desktop/work/vla"
POLICY_PATH="${VLA_ROOT}/weights/pi05_wipes_a800_full_20k"
export PYTHONPATH="${VLA_ROOT}/lerobot/src:${VLA_ROOT}/Alicia-D-SDK:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/home/yunyi/miniconda3/envs/lerobot/bin/python \
  "${VLA_ROOT}/alicia-d-vla-benchmark/scripts/eval_alicia_arms.py" \
  --robot.type=alicia_d_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=alicia_duo_follower_arm \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30},
    "top":   {"type": "opencv", "index_or_path": "/dev/video8", "width": 640, "height": 480, "fps": 30}
  }' \
  --policy.path="${POLICY_PATH}" \
  --dataset_root="${VLA_ROOT}/datasets/wipes-to-green-plate-01" \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.device=cuda \
  --policy.use_amp=true \
  --use_rtc=true \
  --n_action_steps=30 \
  --task="Pick up the pack of cleaning wipes and place it on the green plate" \
  --duration=25 \
  --fps=30 \
  --num_episodes=1 \
  --record_eval=true \
  --eval_dataset_root="${VLA_ROOT}/eval_runs" \
  --eval_dataset_repo_id="local/eval_pi05_smooth" \
  --display_data=false
