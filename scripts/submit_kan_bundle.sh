#!/bin/bash
#SBATCH --job-name=kan_bundle
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=train_kan_bundle_%j.out
#SBATCH --error=train_kan_bundle_%j.err

# Load Environment
module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "Starting KAN Bundle Training"
echo "Script: scripts/train_detector_csv.py"
echo "Model: KAN"
echo "Dataset: unified_dataset (Auto-detecting .pt bundles)"
echo "Date: $(date)"
echo "========================================"

# Run Training
# Note: train_detector_csv.py automatically prefers *_bundle.pt if found in csv_dir
python scripts/train_detector_csv.py \
    --model kan \
    --csv_dir unified_dataset \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --hidden_dim 256 \
    --physics_weight 0.1 \
    --checkpoint_dir models_kan_bundle

echo "========================================"
echo "Training Complete"
echo "Date: $(date)"
echo "========================================"
