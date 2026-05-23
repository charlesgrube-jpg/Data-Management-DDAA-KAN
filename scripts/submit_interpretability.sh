#!/bin/bash
#SBATCH --job-name=ddaa_interpret
#SBATCH --output=logs/interpret_%j.out
#SBATCH --error=logs/interpret_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "========================================"
echo "  DDAA Interpretability Suite"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Start  : $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load miniconda
conda activate ddaa   # adjust to your env name

python scripts/run_interpretability.py \
    --data_dir   unified_dataset \
    --split      test \
    --n_samples  64 \
    --ig_steps   50 \
    --output_dir interpretability_figures \
    --device     auto

echo "Finished: $(date)"
