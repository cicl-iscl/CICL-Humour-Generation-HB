#!/bin/bash
#SBATCH --job-name=Qwen7B_GRPO
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: This script runs smaller Qwen models (7B-14B) on A100 cluster
# For 72B model, use grpo_job_72b.sh on H100 cluster

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

# 3. Define your Project Root
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB

# 4. Activate the UV Environment
source $PROJECT_ROOT/src/.venv/bin/activate
uv sync

# 5. Change to the Project Root
cd $PROJECT_ROOT/src

# Create logs directory if it doesn't exist
mkdir -p logs

# 6. Execute the Training with Accelerate
echo "Starting GRPO training on $SLURM_JOB_NUM_NODES node(s) with 4 A100 GPUs..."
echo "Using Accelerate config: accelerate_config_a100.yaml"

python3 train_qwen_grpo.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "./checkpoints/qwen7b_grpo_no_accelerate" \
    --train_data_file "$PROJECT_ROOT/data/rl_df_train.parquet" \
    --test_data_file "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --joke_rater_model "KonradBRG/joke-rater-xlm-roberta" \
    --num_train_epochs 1

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training job completed successfully."
else
    echo "Training job failed with exit code $?."
fi
