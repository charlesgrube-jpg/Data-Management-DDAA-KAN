#!/bin/bash
#SBATCH --job-name=train_kan
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=train_kan_%j.out
#SBATCH --error=train_kan_%j.err

# SOTA KAN (Kolmogorov-Arnold Network) Training

module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "SOTA KAN Deepfake Detection"
echo "Date: $(date)"
echo "Features:"
echo "  - Custom B-Spline KAN Layers"
echo "  - Wav2Vec2 + ASP Backbone"
echo "  - Differential Learning Rates"
echo "========================================"

python scripts/train_kan.py \
    --epochs 20 \
    --batch_size 48 \
    --freeze_epochs 5 \
    --lr 5e-4 \
    --output_dir models_kan

echo "========================================"
echo "Training Complete"
echo "Date: $(date)"
echo "========================================"
