#!/bin/bash
#SBATCH --job-name=GRPO_Ablation_EN
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# GRPO reward ablation study: train 3 models, each excluding one reward component
# Uses same hyperparams as grpo_en.sh (Qwen2.5-7B, 1 epoch, lr 1e-6, batch 1, grad_accum 8, num_gen 4)

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

COMMON_ARGS="--model_id Qwen/Qwen2.5-7B-Instruct \
    --language en \
    --train_data_file $PROJECT_ROOT/data/rl_df_train.parquet \
    --test_data_file $PROJECT_ROOT/data/rl_df_test.parquet \
    --joke_rater_model KonradBRG/joke-rater-roberta-en \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --generation_batch_size 4 \
    --num_generations 4"

# 4. Ablation 1: No formatting reward
echo "=== Ablation 1/3: Excluding formatting ==="
python3 train_grpo.py $COMMON_ARGS \
    --exclude_rewards formatting \
    --output_dir "./checkpoints/grpo-en-no-formatting" \
    --run_name "grpo-en-no-formatting"
echo "Ablation 1 exit code: $?"

# 5. Ablation 2: No classifier (RoBERTa) reward
echo "=== Ablation 2/3: Excluding roberta_score ==="
python3 train_grpo.py $COMMON_ARGS \
    --exclude_rewards roberta_score \
    --output_dir "./checkpoints/grpo-en-no-classifier" \
    --run_name "grpo-en-no-classifier"
echo "Ablation 2 exit code: $?"

# 6. Ablation 3: No structure diversity reward
echo "=== Ablation 3/3: Excluding structure_diversity ==="
python3 train_grpo.py $COMMON_ARGS \
    --exclude_rewards structure_diversity \
    --output_dir "./checkpoints/grpo-en-no-diversity" \
    --run_name "grpo-en-no-diversity"
echo "Ablation 3 exit code: $?"

echo "All ablation runs completed."
