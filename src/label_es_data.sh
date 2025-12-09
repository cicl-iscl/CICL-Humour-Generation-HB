#!/bin/bash
#SBATCH --job-name=JokeEval_Llama3.1
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: Spanish joke labeling with Llama-3.1-8B-Instruct on 2x A100

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
# Don't override HF_HOME so the library finds the token in ~/.cache/huggingface/token

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Execute the Evaluation Script (simple python with device_map="auto")
echo "Starting Spanish joke evaluation..."

python labeling/llama_eval_es.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --batch_size 8 \
    --output_file "../data/es_data_labeled_llama3.1.csv"

echo "Evaluation finished."
