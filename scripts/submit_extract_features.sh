#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=extract_features_%j.out
#SBATCH --error=extract_features_%j.err

# Load environment
module load miniconda
python_path="$HOME/.conda/envs/ddaa/bin/python"

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "Starting Feature Extraction"
echo "Date: $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================"

# Run extraction
$python_path scripts/preextract_features.py \
    --csv_dir unified_dataset \
    --output_dir unified_dataset/features

echo "========================================"
echo "Extraction Complete"
echo "Date: $(date)"
echo "========================================"
