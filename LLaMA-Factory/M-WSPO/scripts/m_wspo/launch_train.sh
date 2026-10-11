#!/usr/bin/env bash
# Launch m-WSPO training on a multi-tenant GPU box.
#
# Usage:
#   bash scripts/m_wspo/launch_train.sh
#   GPUS=0,1,2,3 bash scripts/m_wspo/launch_train.sh
#   GPUS=7       bash scripts/m_wspo/launch_train.sh   # single-GPU smoke
#
# Optional overrides:
#   YAML=examples/train_m_wspo/<other>.yaml
#   LOG=logs/<custom>.log

set -euo pipefail

cd "$(dirname "$0")/../.."

# ---- which GPUs to use ----
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,5,6,7}"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export NPROC_PER_NODE="$NUM_GPUS"
export FORCE_TORCHRUN=1

# ---- NCCL workarounds for shared / multi-tenant GPUs ----
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0

# ---- Misc ----
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# CONFIG is accepted as an alias for YAML.
YAML="${YAML:-${CONFIG:-}}"
if [[ -z "$YAML" ]]; then
    echo "ERROR: set YAML=<config.yaml> (or CONFIG=<config.yaml>)." >&2
    exit 2
fi
if [[ ! -f "$YAML" ]]; then
    echo "ERROR: config not found: $YAML" >&2
    echo "Available m-WSPO configs:" >&2
    ls examples/train_m_wspo/*.yaml >&2
    exit 2
fi

LOG="${LOG:-logs/m_wspo_full.log}"
mkdir -p "$(dirname "$LOG")"

# The CLI lives in the mwspo311 conda env; activate it if it is not already active.
if ! command -v llamafactory-cli >/dev/null 2>&1; then
    CONDA_SH="/path/to/conda.sh"
    if [[ -f "$CONDA_SH" ]]; then
        # conda.sh references unset vars, so relax strict mode while sourcing.
        set +u
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        # activate your conda/venv environment first
        set -u
    fi
fi
command -v llamafactory-cli >/dev/null 2>&1 || {
    echo "ERROR: llamafactory-cli not on PATH; activate the mwspo311 env first." >&2
    exit 127
}

echo "==== launch_m_wspo ===="
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES  (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "  YAML                 = $YAML"
echo "  LOG                  = $LOG"
echo "======================="

llamafactory-cli train "$YAML" 2>&1 | tee "$LOG"
