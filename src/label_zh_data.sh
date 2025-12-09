#!/bin/bash
#SBATCH --job-name=JokeEval_Qwen7B
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: Chinese joke labeling with Qwen2.5-7B-Instruct on 1x A100

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export HF_HOME=$WORK/cache/huggingface
mkdir -p $HF_HOME

# HuggingFace token (read from file if exists)
export HF_TOKEN=$(cat ~/.huggingface/token 2>/dev/null || echo "")

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Execute the Evaluation Script
echo "Starting Chinese joke evaluation on 1 GPU..."

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    labeling/qwen_eval_zh.py \
    --model_name "Orion-zhen/Qwen2.5-7B-Instruct-Uncensored" \
    --batch_size 8 \
    --output_file "../data/zh_data_labeled_qwen7b.csv"

echo "Evaluation finished."
