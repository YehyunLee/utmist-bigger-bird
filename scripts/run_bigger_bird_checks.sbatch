#!/bin/bash
#SBATCH --job-name=bigger-bird-v2-check
#SBATCH --account=def-guerzhoy
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/%u/bigger_bird_check_%j.out

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
echo BIGGER_BIRD_FINAL_CHECKS_COMPLETE
