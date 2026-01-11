#!/bin/bash
#SBATCH --job-name=RoBERTa_ES
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Train Spanish RoBERTa joke rater on es_data_labeled_llama3.1.csv
# Single GPU, no accelerate

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
uv sync
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Execute Training (single GPU, no accelerate)
echo "Starting Spanish RoBERTa training on 1 GPU..."

python3 train_roberta.py \
    --model_name "FacebookAI/xlm-roberta-large" \
    --language "es" \
    --data_dir "$PROJECT_ROOT/data" \
    --output_dir "./checkpoints/roberta-es" \
    --hub_model_id "KonradBRG/joke-rater-roberta-es" \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 32

# Check exit status
if [ $? -eq 0 ]; then
    echo "Spanish RoBERTa training completed successfully."
else
    echo "Spanish RoBERTa training failed with exit code $?."
fi
