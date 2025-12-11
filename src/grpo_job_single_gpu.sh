#!/bin/bash
#SBATCH --job-name=Qwen3B_GRPO_1GPU
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: Single GPU training for Qwen 3B - no Accelerate needed

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 3. Execute Training directly (no Accelerate)
echo "Starting single-GPU GRPO training..."

python train_qwen_grpo.py \
    --model_id "Qwen/Qwen2.5-3B-Instruct" \
    --output_dir "./checkpoints/qwen3b_grpo_1gpu" \
    --train_data_file "$PROJECT_ROOT/data/rl_df_train.parquet" \
    --test_data_file "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --joke_rater_model "KonradBRG/joke-rater-xlm-roberta" \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --generation_batch_size 4 \
    --num_generations 4 \
    --max_completion_length 48

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training job completed successfully."
else
    echo "Training job failed with exit code $?."
fi
