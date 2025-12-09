#!/bin/bash
#SBATCH --job-name=JokeEval_Qwen32B_ZH
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: Chinese joke labeling with Qwen2.5-32B-Instruct on 2x A100
# 32B needs ~64GB VRAM, device_map="auto" spreads across GPUs

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 3. Execute
echo "Starting Chinese joke evaluation with Qwen2.5-32B..."

python labeling/qwen_eval_zh.py \
    --model_name "Qwen/Qwen2.5-32B-Instruct" \
    --batch_size 2 \
    --output_file "../data/zh_data_labeled_qwen32b.csv"

echo "Evaluation finished."
