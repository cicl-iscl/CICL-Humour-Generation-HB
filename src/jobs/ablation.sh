#!/bin/bash
#SBATCH --job-name=Ablation
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Ablation study for all languages (en, zh, es)
# Compares xlm-roberta-large and mdeberta-v3-base
# With frozen backbone vs full fine-tuning

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Run ablation study for all languages
echo "Starting ablation study for all languages..."

python3 ablation_study.py \
    --language all \
    --data_dir "$PROJECT_ROOT/data" \
    --output_dir "./ablation_results" \
    --num_epochs 10 \
    --num_epochs_frozen 20 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --learning_rate_frozen 1e-3

# Check exit status
if [ $? -eq 0 ]; then
    echo "Ablation study completed successfully."
else
    echo "Ablation study failed with exit code $?."
fi
