#!/bin/bash
#SBATCH --job-name=Model_Eval
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: This script evaluates and compares multiple trained models
# Uses a single A100 GPU since we evaluate one model at a time

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs eval_results

# 3. Define models to compare
# Add your trained model paths here after training completes
# Example paths - adjust based on your actual checkpoint locations

MODELS=(
    # Base models (for comparison)
    "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # Trained models (uncomment after training)
    "./checkpoints/qwen3b_grpo",
    # "./checkpoints/qwen7b_grpo"
    # "./checkpoints/qwen32b_grpo"
    # "./checkpoints/deepseek_r1_32b_grpo"
)

MODEL_NAMES=(
    "Qwen-3B-Base"
    # "Qwen-32B-Base"
    "Qwen-3B-GRPO"
    # "Qwen-7B-GRPO"
    # "Qwen-32B-GRPO"
    # "DeepSeek-R1-32B-GRPO"
)

# 4. Run evaluation
echo "Starting model evaluation..."
echo "Comparing ${#MODELS[@]} models"

python3 evaluate_models.py \
    --models "${MODELS[@]}" \
    --model_names "${MODEL_NAMES[@]}" \
    --test_data "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --output_dir "./eval_results" \
    --num_generations 3 \
    --temperature 0.7 \
    --batch_size 8

# Check exit status
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    echo "Results saved to ./eval_results/"
else
    echo "Evaluation failed with exit code $?."
fi
