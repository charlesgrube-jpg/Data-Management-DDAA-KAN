#!/bin/bash
#SBATCH --job-name=extract_ssl
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=extract_ssl_%j.out
#SBATCH --error=extract_ssl_%j.err

# Extract SSL features using Wav2Vec2/WavLM on GPU

module load miniconda
source ~/.bashrc
conda activate ddaa

cd ~/project/Data-Management-DDAA-KAN

echo "========================================"
echo "SSL Feature Extraction"
echo "Date: $(date)"
echo "========================================"

# Fix package versions for compatibility
echo "Upgrading packages for compatibility..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
pip install --upgrade transformers 2>/dev/null || true

echo "Packages updated. Starting extraction..."
echo "Model: wav2vec2-base"
echo "========================================"

# Run extraction
python scripts/preextract_ssl_features.py \
    --model wav2vec2-base \
    --device cuda \
    --csv_dir unified_dataset

echo "========================================"
echo "SSL Extraction Complete"
echo "Date: $(date)"
echo "========================================"
