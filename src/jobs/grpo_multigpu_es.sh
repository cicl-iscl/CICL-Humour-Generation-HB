#!/bin/bash
#SBATCH --job-name=GRPO_ES_Multi
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Train Spanish GRPO joke generator using Qwen2.5-7B
# Multi-GPU with accelerate (4 GPUs)

######################
### Set environment ###
######################

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export GPUS_PER_NODE=4
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
export VLLM_CONFIGURE_LOGGING=0
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
uv sync
cd $PROJECT_ROOT/src
mkdir -p logs

######################
### Training Script ###
######################

SCRIPT="train_grpo_multi.py"
SCRIPT_ARGS=" \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --language es \
    --output_dir ./checkpoints/grpo-es-multi \
    --train_data_file $PROJECT_ROOT/data/rl_df_train_es.parquet \
    --test_data_file $PROJECT_ROOT/data/rl_df_test_es.parquet \
    --joke_rater_model KonradBRG/joke-rater-roberta-es \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --generation_batch_size 4 \
    --num_generations 4 \
    "

# 4. Execute Training (multi-GPU with accelerate)
echo "Starting Spanish GRPO training on $GPUS_PER_NODE GPUs with accelerate..."

accelerate launch --num_processes $GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo "Spanish GRPO multi-GPU training completed successfully."
else
    echo "Spanish GRPO multi-GPU training failed with exit code $?."
fi
