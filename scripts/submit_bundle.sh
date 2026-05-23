#!/bin/bash
#SBATCH --job-name=bundle_ssl
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=bundle_ssl_%j.out
#SBATCH --error=bundle_ssl_%j.err

# Bundle SSL .npy features into single .pt files for instant loading

module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "Bundling SSL Features into Single Files"
echo "Date: $(date)"
echo "========================================"

python scripts/bundle_ssl_features.py

echo "========================================"
echo "Bundling Complete"
echo "Date: $(date)"
echo "========================================"
echo ""
echo "Now run: sbatch scripts/submit_train_all.sh"
