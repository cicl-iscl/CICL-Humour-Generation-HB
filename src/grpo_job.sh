#!/bin/bash
#SBATCH --job-name=Qwen72B_GRPO
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: --mem flags are omitted per bwUniCluster documentation.

# 1. Load Modules (Ensure you load the exact versions available)
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

export ACCELERATE_USE_NCCL=1
export NCCL_ASYNC_INIT=0
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

# 2. Define your Project Root
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB

# 3. Activate the UV Environment
source $PROJECT_ROOT/src/.venv/bin/activate

# 4. Change to the Project Root (Ensures script runs from the correct location)
cd $PROJECT_ROOT
cd src

# 5. Execute the Training (using 4 processes for 4 GPUs)
echo "Starting distributed training on $SLURM_JOB_NUM_NODES node(s) with 4 GPUs..."

accelerate launch --num_processes 2 train_qwen_grpo.py \
    --model_id "Qwen/Qwen2.5-72B-Instruct" \
    --output_dir "./checkpoints/qwen72b_grpo_run_01" \
    --train_data_file "$PROJECT_ROOT/data/rl_df_train.parquet" \
    --test_data_file "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 4
