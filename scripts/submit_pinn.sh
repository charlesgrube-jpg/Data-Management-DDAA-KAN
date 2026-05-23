#!/bin/bash
#SBATCH --job-name=train_pinn
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=train_pinn_%j.out
#SBATCH --error=train_pinn_%j.err

# SOTA Physics-Informed (PINN) Deepfake Detection

module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "SOTA PINN Deepfake Detection"
echo "Date: $(date)"
echo "Features:"
echo "  - Physics Loss (Temporal Smoothness)"
echo "  - Enforcing Spectral Continuity"
echo "  - Wav2Vec2 + ASP Backbone"
echo "========================================"

python scripts/train_pinn.py \
    --epochs 20 \
    --batch_size 48 \
    --freeze_epochs 5 \
    --lr 5e-4 \
    --physics_weight 0.1 \
    --output_dir models_pinn

echo "========================================"
echo "Training Complete"
echo "Date: $(date)"
echo "========================================"
