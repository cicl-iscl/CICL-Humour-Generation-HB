#!/bin/bash
#SBATCH --job-name=Eval_Ablation
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

# Evaluate GRPO reward ablation models (full + 3 ablations)

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

# 4. Run ablation evaluation
echo "Starting ablation evaluation..."

python3 evaluate_ablation.py \
    --test_data "$PROJECT_ROOT/data/rl_df_test.parquet" \
    --output_dir "./ablation_results" \
    --language en \
    --num_generations 3 \
    --max_new_tokens 128 \
    --temperature 0.8 \
    --batch_size 4 \
    --reward_batch_size 32

if [ $? -eq 0 ]; then
    echo "Ablation evaluation completed successfully."
else
    echo "Ablation evaluation failed with exit code $?."
fi
