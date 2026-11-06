#!/usr/bin/env bash
# ACT-style temporal ensemble; CKPT_PATH overrides the default warm_full checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_REPO_ID="${EVAL_REPO_ID:-local/eval_turbovla_act_ensemble_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/eval_turbovla_warm_full_smooth.sh" \
  --temporal_ensemble_mode=act \
  --n_action_steps=1 \
  --eval_dataset_repo_id="${EVAL_REPO_ID}" \
  "$@"
