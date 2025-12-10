#!/bin/bash
#SBATCH --job-name=JokeRater_Training
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NOTE: This script trains the XLM-RoBERTa joke rater on 4x A100

# 1. Load Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Environment Variables
export ACCELERATE_USE_NCCL=1
export NCCL_ASYNC_INIT=0
export NCCL_DEBUG=WARN
export TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# 3. Project Setup
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB
source $PROJECT_ROOT/src/.venv/bin/activate
cd $PROJECT_ROOT/src
mkdir -p logs

# 4. Execute Training with Accelerate
echo "Starting distributed training on 1 node with 4 A100 GPUs using accelerate..."

accelerate launch --config_file accelerate_config_a100.yaml \
    train_multilingual_roberta_model.py \
    --model_name "xlm-roberta-large" \
    --output_dir "./results/joke_rater" \
    --hub_model_id "KonradBRG/joke-rater-xlm-roberta" \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 32

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training job completed successfully."
else
    echo "Training job failed with exit code $?."
fi
