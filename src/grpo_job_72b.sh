#!/bin/bash
#SBATCH --job-name=Qwen72B_GRPO
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

export ACCELERATE_USE_NCCL=1
export NCCL_ASYNC_INIT=0
export NCCL_DEBUG=INFO
export HF_HOME=$WORK/cache/huggingface
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p $HF_HOME $TORCH_EXTENSIONS_DIR

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate

cd $PROJECT_ROOT
cd src
echo "Starting ZeRO-3 Distributed Training on $SLURM_JOB_NUM_NODES node with 4 GPUs..."

accelerate launch --config_file accelerate_config_72b.yaml \
    train_qwen_grpo_72b.py \
    --model_id "Qwen/Qwen2.5-72B-Instruct" \
    --output_dir "./checkpoints/qwen72b_grpo_run_01" \
    --train_data_file "$PROJECT_ROOT/data/rl_df_train.parquet" \
    --test_data_file "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --generation_batch_size 1 \
    --max_completion_length 1024