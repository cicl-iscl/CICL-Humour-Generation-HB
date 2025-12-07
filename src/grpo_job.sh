#!/bin/bash
#SBATCH --job-name=grpo_humor
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: Memory flags (--mem) are omitted as per cluster documentation instructions.

# 1. Load Modules
module load cuda/12.1
module load miniconda/3

# 2. Activate Environment
source ~/.venv/bin/activate

# 3. Debugging Info
echo "Job running on node: $SLURMD_NODENAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# 4. Run the Training using Accelerate Launch
# Must launch 4 processes to match the 4 GPUs.
accelerate launch --num_processes 4 run_grpo_training.py \
    --model_id "Qwen/Qwen2.5-72B-Instruct" \
    --output_dir "./checkpoints/qwen72b_grpo_run_01" \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 4