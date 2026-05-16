#!/bin/bash
#SBATCH --job-name=train_all_models
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --output=train_%a_%A.out
#SBATCH --error=train_%a_%A.err

# Train All 4 Defense Models
# Array job: 0=transformer, 1=kan, 2=neural_ode, 3=pinn

module load miniconda
PYTHON_PATH="$HOME/.conda/envs/ddaa/bin/python"

cd ~/project/Data-Management-DDAA-KAN

# Model name mapping
MODELS=("transformer" "kan" "neural_ode" "pinn")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Training Model: $MODEL"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"
echo "========================================"

# Training parameters
EPOCHS=200
BATCH_SIZE=512
HIDDEN_DIM=256
LR=0.0001

# Run training
$PYTHON_PATH scripts/train_detector_csv.py \
    --model $MODEL \
    --csv_dir unified_dataset \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --lr $LR \
    --num_workers 0 \
    --checkpoint_dir checkpoints

echo "========================================"
echo "Training Complete: $MODEL"
echo "Date: $(date)"
echo "========================================"
