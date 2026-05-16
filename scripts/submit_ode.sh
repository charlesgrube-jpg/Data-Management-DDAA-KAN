#!/bin/bash
#SBATCH --job-name=train_ode
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=train_ode_%j.out
#SBATCH --error=train_ode_%j.err

# SOTA Neural ODE Deepfake Detection

module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "SOTA Neural ODE Deepfake Detection"
echo "Date: $(date)"
echo "Features:"
echo "  - Latent Dynamics (Neural ODE Head)"
echo "  - Adaptive Step Solver (or RK4 fallback)"
echo "  - Wav2Vec2 + ASP Backbone"
echo "========================================"

# Attempt to ensure torchdiffeq is present (optional)
pip install torchdiffeq > /dev/null 2>&1

python scripts/train_ode.py \
    --epochs 20 \
    --batch_size 16 \
    --freeze_epochs 5 \
    --lr 1e-4 \
    --output_dir models_ode

echo "========================================"
echo "Training Complete"
echo "Date: $(date)"
echo "========================================"
