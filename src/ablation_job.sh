#!/bin/bash
#SBATCH --job-name=JokeRater_Ablation
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: Ablation study comparing xlm-roberta-large vs mdeberta-v3-base
# with and without fine-tuning on the joke rating task

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs ablation_results

# 3. Run ablation study
echo "Starting ablation study..."

python ablation_study.py \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --output_dir "./ablation_results"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Ablation study completed successfully."
    echo "Results saved to ./ablation_results/"
else
    echo "Ablation study failed with exit code $?."
fi
