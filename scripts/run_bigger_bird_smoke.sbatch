#!/bin/bash
#SBATCH --job-name=bigger-bird-v2-smoke
#SBATCH --account=def-guerzhoy
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/bigger_bird_v2_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 arrow
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH=.
export HF_HOME=/scratch/thomas7/hf-cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python tests/test_bigger_bird_flash.py
python tests/test_bigger_bird_cache.py
python scripts/benchmark_bigger_bird_flash.py --seqs 4096,8192,16384,32768 --iters 3 --block-m 32 --output benchmarks/bigger_bird_flash/microbench_fused32.json
python scripts/benchmark_bigger_bird_flash.py --seqs 4096,8192,16384,32768,65536,131072 --iters 3 --block-m 64 --output benchmarks/bigger_bird_flash/microbench_fused64.json
BB_BLOCK_M=$(python -c 'import json; a=json.load(open("benchmarks/bigger_bird_flash/microbench_fused32.json"))["results"]; b=json.load(open("benchmarks/bigger_bird_flash/microbench_fused64.json"))["results"]; score=lambda rows:sum(r["sparse_complete_ms"] for r in rows if r["n"] in (8192,16384)); print(64 if score(b)<score(a) else 32)')
echo "Selected query tile size: ${BB_BLOCK_M}"
python scripts/smoke_bigger_bird_flash.py --seqs 8192,16384,16521 --depths 0.25,0.75 --modes last_query,causal_chunk --route-chunk 4096 --block-m "${BB_BLOCK_M}" --cached-dense --cached-sparse --only-cached --output benchmarks/bigger_bird_flash/smoke_fused_cached.json
echo BIGGER_BIRD_BATCH_COMPLETE
