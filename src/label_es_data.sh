#!/bin/bash
#SBATCH --job-name=JokeEval_Llama3.1
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00 # Reduce time limit since it's only inference
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# 1. Load Modules (Ensure you load the exact versions available)
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"

# 2. Define your Project Root
# NOTE: Update this path to where your evaluation script and data are located
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/CICL-Humour-Generation-HB

# 3. Activate the UV Environment
source $PROJECT_ROOT/src/.venv/bin/activate

# 4. Change to the Directory Containing the Evaluation Script
cd $PROJECT_ROOT
cd src

# 5. Execute the Evaluation Script
# We use 'accelerate launch' with --num_processes 1 because inference is typically
# limited by VRAM and 1 process is simpler and efficient for single-GPU inference.
# If you decide to use multiple GPUs for faster execution, change --num_processes
# and --gres=gpu:<N>.

echo "Starting joke evaluation using accelerate on 1 GPU..."

accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    labeling/llama_eval_es.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --batch_size 16 \
    --output_file "../data/es_data_labeled_llama3.1.csv"

echo "Evaluation finished."