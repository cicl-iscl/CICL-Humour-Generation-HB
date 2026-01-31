#!/bin/bash
#SBATCH --job-name=Compare_Base_vs_Trained
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Compare base Qwen2.5-7B-Instruct vs GRPO-trained models
# Generates jokes on test set and compares mean reward scores

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
uv sync
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Run comparison for all languages
echo "Starting base vs trained model comparison..."

python3 compare_base_vs_trained.py \
    --languages en zh es \
    --data_dir "$PROJECT_ROOT/data" \
    --output_dir "./comparison_results" \
    --max_new_tokens 128 \
    --temperature 0.7 \
    --batch_size 4

if [ $? -eq 0 ]; then
    echo "Comparison completed successfully."
else
    echo "Comparison failed with exit code $?."
fi
