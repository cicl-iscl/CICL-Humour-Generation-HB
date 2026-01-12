#!/bin/bash
#SBATCH --job-name=Submit_ES
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Generate Spanish joke submissions

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1

# 2. Environment Variables
export CUDA_VISIBLE_DEVICES=0

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Run submission generation
echo "Generating Spanish submissions..."

python3 run_submissions.py \
    --input_tsv "$PROJECT_ROOT/data/task-a-es.tsv" \
    --model_id "KonradBRG/Qwen2.5-7B-Instruct-Jokester-Spanish" \
    --language "es" \
    --output_tsv "$PROJECT_ROOT/submissions/task-a-es-submission.tsv" \
    --max_new_tokens 128 \
    --temperature 0.8

echo "Done!"
