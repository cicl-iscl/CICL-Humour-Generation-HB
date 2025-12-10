#!/bin/bash
#SBATCH --job-name=Qwen32B_GRPO
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: This script runs Qwen2.5-32B on A100 cluster with FSDP
# 32B model requires ~64GB VRAM in bf16, FSDP shards across 4x A100 (320GB total)

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables for Distributed Training
export ACCELERATE_USE_NCCL=1
export NCCL_ASYNC_INIT=0
export NCCL_DEBUG=WARN
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# Wandb API key (run `wandb login` once on the cluster to set this up)
export WANDB_API_KEY=$(cat ~/.netrc 2>/dev/null | grep -A2 "api.wandb.ai" | grep password | awk '{print $2}' || echo "")

# Memory optimization for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Execute the Training with FSDP via Accelerate
echo "Starting FSDP Training on $SLURM_JOB_NUM_NODES node with 4 A100 GPUs..."
echo "Using Accelerate config: accelerate_config_fsdp_32b.yaml"

accelerate launch --config_file accelerate_config_fsdp_32b.yaml \
    train_qwen_grpo_32b.py \
    --model_id "Qwen/Qwen2.5-32B-Instruct" \
    --output_dir "./checkpoints/qwen32b_grpo" \
    --train_data_file "$PROJECT_ROOT/data/rl_df_train.parquet" \
    --test_data_file "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --num_train_epochs 1 \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --generation_batch_size 4 \
    --num_generations 4 \
    --max_completion_length 64

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training job completed successfully."
else
    echo "Training job failed with exit code $?."
fi
