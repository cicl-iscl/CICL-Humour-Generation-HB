#!/bin/bash
#SBATCH --job-name=GRPO_EN
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Train English GRPO joke generator using Qwen2.5-7B
# Single GPU, no accelerate

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
export VLLM_CONFIGURE_LOGGING=0
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
uv sync
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Execute Training (single GPU, no accelerate)
echo "Starting English GRPO training on 1 GPU..."

python3 train_grpo.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --language "en" \
    --output_dir "./checkpoints/grpo-en" \
    --train_data_file "$PROJECT_ROOT/data/rl_df_train.parquet" \
    --test_data_file "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --joke_rater_model "KonradBRG/joke-rater-roberta-en" \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --generation_batch_size 4 \
    --num_generations 4

# Check exit status
if [ $? -eq 0 ]; then
    echo "English GRPO training completed successfully."
else
    echo "English GRPO training failed with exit code $?."
fi
