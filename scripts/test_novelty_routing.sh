#!/bin/bash
#SBATCH --account=def-guerzhoy_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre07/scratch/%u/exp18_novelty_test_%j.out

set -euo pipefail

ROOT="/lustre06/project/6087692/${USER}/utmist-bigger-bird"
SCRATCH_DIR="/lustre07/scratch/${USER}"

module load StdEnv/2023 gcc arrow cuda/12.6 python/3.11.5 2>/dev/null || true
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH_DIR}/hf-cache"
export SCRATCH="${SCRATCH_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT}"

# Scale-invariant hybrid routing: budget = ratio * seq_len (no fixed top_k)
# 2K @ 5% = 102 tokens, 4K @ 5% = 204, 8K @ 5% = 410

for SEQ in 2048 4096 8192; do
  for RATIO in 0.05 0.10; do
    echo "=== Hybrid routing at ${SEQ}, ratio=${RATIO} ==="
    ATTN_KWARGS=$(python3 - <<PY
import json
print(json.dumps({
    "top_k": 2048,
    "low_rank_dim": 128,
    "window_size": 256,
    "gate_threshold": 0.5,
    "peak_threshold": -1.0,
    "linear_weight": 0.5,
    "use_triton": False,
    "always_global": True,
    "num_route_queries": 4,
    "adaptive_low_rank": True,
    "routing_mode": "hybrid",
    "novelty_ratio": ${RATIO},
    "novelty_window": 64,
}))
PY
)
    python -m eval.ruler_llama.run_generative \
      --task niah --exp 18 --seq "${SEQ}" --depth 0.5 \
      --eval-samples 128 --max-examples 10 \
      --attn-kwargs "${ATTN_KWARGS}" || true
  done
done
