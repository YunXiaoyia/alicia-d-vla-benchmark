#!/usr/bin/env bash
# ==============================================================================
# [StarVLA - Best Variant: FixQwen 10k] (Frozen Full Qwen3.5-2B-VL + Flow-Matching DiT-B / QwenGR00T)
# 模式：RecedingHorizonTrajectoryBlender 五次多项式 C² 边界锚定 + 块内高斯平滑
# 硬件端口：机械臂 /dev/ttyACM0 | 全局相机 top=/dev/video8 | 腕部相机 wrist=/dev/video6
# ==============================================================================
set -euo pipefail

VLA_ROOT="/media/yunyi/Data/Desktop/work/vla"
export PYTHONPATH="${VLA_ROOT}/starVLA:${VLA_ROOT}/lerobot/src:${VLA_ROOT}/Alicia-D-SDK:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_PATH="${CKPT_PATH:-${VLA_ROOT}/weights/starvla_wipes_qwen3_5_2b_gr00t_fixqwen/pytorch_model.pt}"

/home/yunyi/miniconda3/envs/starvla/bin/python \
  "${VLA_ROOT}/alicia-d-vla-benchmark/scripts/eval_alicia_starvla.py" \
  --ckpt_path="${CKPT_PATH}" \
  --robot.port=/dev/ttyACM0 \
  --robot.id=alicia_duo_follower_arm \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30},
    "top":   {"type": "opencv", "index_or_path": "/dev/video8", "width": 640, "height": 480, "fps": 30}
  }' \
  --n_action_steps=30 \
  --temporal_ensemble_coeff=0.05 \
  --task="Pick up the pack of cleaning wipes and place it on the green plate" \
  --duration=25 \
  --fps=30 \
  --num_episodes=1 \
  --record_eval=true \
  --eval_dataset_root="${VLA_ROOT}/eval_runs" \
  --eval_dataset_repo_id="local/eval_starvla_fixqwen_smooth" \
  --display_data=false \
  "$@"
