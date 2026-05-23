#!/bin/bash
#SBATCH --job-name=regen_figs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN/logs/regen_figs_%j.out
#SBATCH --error=/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN/logs/regen_figs_%j.err

echo "regen_figs | Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"

source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate ddaa

cd /gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN

python scripts/regen_figs_correct.py

echo "Done: $(date)"
