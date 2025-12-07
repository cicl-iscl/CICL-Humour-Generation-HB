#!/bin/bash
#SBATCH --job-name=JokeRater_Training
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de


module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

export ACCELERATE_USE_NCCL=1
export NCCL_ASYNC_INIT=0
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

# Define Project Root
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB

# Change to the Project Root
cd $PROJECT_ROOT

# Activate the UV Environment (Must be done after module loading)
source $PROJECT_ROOT/src/.venv/bin/activate
echo "Starting distributed training on 1 node(s) with 4 GPUs using accelerate..."

accelerate launch \
    --num_processes 4 \
    train_multilingual_roberta_model.py \
    --model_name "xlm-roberta-large" \
    --output_dir "./results/joke_rater" \
    --hub_model_id "KonradBRG/joke-rater-xlm-roberta" \
    --num_train_epochs 200 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 64

# Check the exit status of the training script
if [ $? -eq 0 ]; then
    echo "Training job completed successfully."
else
    echo "Training job failed with exit code $?."
fi