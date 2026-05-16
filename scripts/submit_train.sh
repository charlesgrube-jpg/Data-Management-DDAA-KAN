#!/bin/bash
#SBATCH --job-name=train_hp
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=train_hp_%j.out
#SBATCH --error=train_hp_%j.err

# High-Performance Deepfake Detection Training

module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "High-Performance Deepfake Detection"
echo "Date: $(date)"
echo "Features:"
echo "  - Multi-layer hidden state fusion"
echo "  - Attentive Statistics Pooling"
echo "  - Depthwise separable downsampling"
echo "  - SpecAugment on features"
echo "  - Differential Learning Rates (1e-5 backbone)"
echo "========================================"

python scripts/train.py \
    --epochs 20 \
    --batch_size 48 \
    --freeze_epochs 5 \
    --lr 5e-4 \
    --csv_dir unified_dataset \
    --output_dir models

echo "========================================"
echo "Training Complete"
echo "Date: $(date)"
echo "========================================"
