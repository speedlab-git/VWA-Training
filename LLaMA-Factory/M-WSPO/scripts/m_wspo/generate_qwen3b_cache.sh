#!/usr/bin/env bash
# Generate the weak log-probability cache using Qwen2.5-VL-3B as the weak pair.
#
# Usage: GPU=7 bash scripts/m_wspo/generate_qwen3b_cache.sh
#
# Prerequisites:
#   - Qwen2.5-VL-3B-Instruct downloaded to models/Qwen2.5-VL-3B-Instruct
#   - DPO-finetuned weak model merged at saves/qwen2_vl_3b/dpo_rlhfv_merged
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_SH="/path/to/conda.sh"
if ! command -v llamafactory-cli >/dev/null 2>&1 && [[ -f "$CONDA_SH" ]]; then
    set +u; source "$CONDA_SH"; # activate your conda/venv environment first; set -u
fi

export CUDA_VISIBLE_DEVICES="${GPU:-7}"

WEAK_REF=Qwen/Qwen2.5-VL-3B-Instruct
WEAK_ALN=saves/qwen2_vl_3b/dpo_rlhfv_merged
OUTPUT=cache/rlhf_v_mwspo_weak_lp_qwen3b.parquet

echo "=== Generating Qwen2.5-VL-3B weak cache ==="
echo "  weak_ref     : $WEAK_REF"
echo "  weak_aligned : $WEAK_ALN"
echo "  output       : $OUTPUT"
echo "  GPU          : $CUDA_VISIBLE_DEVICES"
echo "==========================================="

mkdir -p cache

CUDA_VISIBLE_DEVICES=7 python scripts/m_wspo/cache_weak_logprobs.py \
    --dataset         rlhf_v_mwspo \
    --weak_ref_model  "$WEAK_REF" \
    --weak_aligned_model "$WEAK_ALN" \
    --template_model_path "$WEAK_REF" \
    --template        qwen2_vl \
    --cutoff_len      2048 \
    --batch_size      4 \
    --output          "$OUTPUT" \
    --bf16

echo "Done. Cache written to $OUTPUT"
