#!/bin/bash
#SBATCH --job-name=gen_ddaa_7gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load modules (adjust based on Grace/Bouchet availability)
module load miniconda
module load cuda/11.8

# Activate environment
conda activate ddaa

echo "Job started on $(hostname) at $(date)"
echo "Workdir: $(pwd)"

# Ensure data is present
if [ ! -d "mozilla_cv_data/cv-corpus-24.0-2025-12-05/en" ]; then
    echo "Error: Mozilla CV data not found in mozilla_cv_data/"
    exit 1
fi

# Run pipeline
python run_pipeline.py

echo "Job finished at $(date)"
