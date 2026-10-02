#!/bin/bash
#SBATCH --job-name=bb-rd3-probe
#SBATCH --account=def-guerzhoy
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/bb_round3_probe_%j.out
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 arrow
cd "${SLURM_SUBMIT_DIR}"
source /scratch/thomas7/r1-venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}
export HF_HOME=/scratch/thomas7/hf-cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/scratch/thomas7/triton-bb-round2-20260928
mkdir -p "${TRITON_CACHE_DIR}"
nvidia-smi -L
python scripts/run_round3_probe.py
