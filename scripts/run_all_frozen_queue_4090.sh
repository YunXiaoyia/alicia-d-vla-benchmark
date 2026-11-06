#!/usr/bin/env bash
# ==============================================================================
# 4090 串行后台训练队列 (Sequential Training Queue on 1x RTX 4090)
# ------------------------------------------------------------------------------
# 执行顺序:
#   Job 1: TurboVLA (Frozen DINOv3 + BERT, LIBERO 40k init, BS=64, 8,000 steps)
#   Job 2: StarVLA - FixQwen (Frozen full Qwen3.5-2B VLM, train DiT-B 155M, BS=4x8=32, 10,000 steps)
#   Job 3: StarVLA - FreezeViT (Frozen Qwen3.5-VL visual ViT, train LLM@1e-5 + DiT-B@1e-4, BS=1x32=32, 10,000 steps)
# ==============================================================================

mkdir -p /home/yunyi/Desktop/vla/logs
mkdir -p /home/yunyi/Desktop/vla/TurboVLA/logs
mkdir -p /home/yunyi/Desktop/vla/starVLA/playground/Checkpoints

QUEUE_LOG="/home/yunyi/Desktop/vla/logs/queue_frozen_3jobs.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

log_msg "=================================================================="
log_msg "Starting 3-Job Sequential Frozen-Backbone Training Queue on 4090"
log_msg "=================================================================="

# ------------------------------------------------------------------------------
# Job 1: TurboVLA (Frozen DINOv3)
# ------------------------------------------------------------------------------
TURBO_LOG="/home/yunyi/Desktop/vla/TurboVLA/logs/train_wipes_freeze_vit.log"
log_msg "[Job 1/3] Starting TurboVLA (Frozen DINOv3, 8000 steps) -> ${TURBO_LOG}"
bash /home/yunyi/Desktop/vla/scripts/train_turbovla_wipes_freeze_vit.sh > "${TURBO_LOG}" 2>&1
RC1=$?
log_msg "[Job 1/3] Finished TurboVLA with exit code ${RC1}"
sleep 5

# ------------------------------------------------------------------------------
# Job 2: StarVLA - FixQwen (Freeze entire Qwen3.5-2B VLM)
# ------------------------------------------------------------------------------
FIXQWEN_LOG="/home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_fixqwen.log"
log_msg "[Job 2/3] Starting StarVLA FixQwen (Freeze full Qwen3.5-2B, 10000 steps) -> ${FIXQWEN_LOG}"
bash /home/yunyi/Desktop/vla/starVLA/examples/realRobots/Alicia_D_wipes/run_wipes_fixqwen_4090.sh > "${FIXQWEN_LOG}" 2>&1
RC2=$?
log_msg "[Job 2/3] Finished StarVLA FixQwen with exit code ${RC2}"
sleep 5

# ------------------------------------------------------------------------------
# Job 3: StarVLA - FreezeViT (Freeze only Qwen3.5-VL visual tower)
# ------------------------------------------------------------------------------
FREEZEVIT_LOG="/home/yunyi/Desktop/vla/starVLA/playground/Checkpoints/wipes_freeze_vit.log"
log_msg "[Job 3/3] Starting StarVLA FreezeViT (Freeze visual ViT only, 10000 steps) -> ${FREEZEVIT_LOG}"
bash /home/yunyi/Desktop/vla/starVLA/examples/realRobots/Alicia_D_wipes/run_wipes_freeze_vit_4090.sh > "${FREEZEVIT_LOG}" 2>&1
RC3=$?
log_msg "[Job 3/3] Finished StarVLA FreezeViT with exit code ${RC3}"

log_msg "=================================================================="
log_msg "All 3 Jobs Completed! Exit codes: Job1=${RC1}, Job2=${RC2}, Job3=${RC3}"
log_msg "=================================================================="
