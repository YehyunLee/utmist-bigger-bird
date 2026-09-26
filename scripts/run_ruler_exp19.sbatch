#!/bin/bash
#SBATCH --job-name=ruler-exp19-4k-16k
#SBATCH --account=def-guerzhoy
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/ruler_exp19_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 arrow
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}
export HF_HOME=/scratch/thomas7/hf-cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seq in 4096 16384; do
    echo "RULER_EXP19_START seq=${seq} job=${SLURM_JOB_ID}"
    python -m eval.ruler_llama.run_generative \
        --task niah --exp 19 --seq "${seq}" --depth 0.5 \
        --eval-samples 128 --max-examples 30
    echo "RULER_EXP19_DONE seq=${seq}"
done
python scripts/build_dashboard.py
echo RULER_EXP19_BATCH_COMPLETE
