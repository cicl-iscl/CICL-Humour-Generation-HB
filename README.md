# MWAHAHA — Competition on Humor Generation
Hou & Brüggemann

This repository contains our work on computational humour generation for the [MWAHAHA Competition](https://pln-fing-udelar.github.io/semeval-2026-humor-gen/).
Our goal is to develop a reinforcement-learning–based humour generation system.

# Overview
We experiment with GRPO-based Reinforcement Learning (potentially with **Qwen-VL** as a base model) and combine multiple reward signals:

- Humor score (classifier):	0–10 rating from a trained funniness classifier
- Text heuristics	e.g., favor short jokes, sentiment shifts / punchlines
- Rejection model: penalizes outputs that are not jokes

# Repository Structure
```
├── notebooks/   # Prototyping, data processing, model training experiments
├── data/        # Joke datasets, generated ratings, evaluation artifacts
├── src/         # Training scripts and job files
│   ├── jobs/    # SLURM job scripts for each language
│   ├── grpo/    # GRPO reward functions and CLI
│   ├── joke_rater/  # RoBERTa joke rater model and preprocessing
│   └── labeling/    # LLM-based data labeling scripts
└── humor-archi.drawio  # System architecture diagram
```

# Training

## Prerequisites

1. Set up the environment:
```bash
cd src
uv sync
source .venv/bin/activate
```

2. Prepare GRPO training data (for each language):
```bash
python prepare_grpo_data.py --language en --test_tsv ../task-a-en.tsv
python prepare_grpo_data.py --language zh --test_tsv ../task-a-zh.tsv
python prepare_grpo_data.py --language es --test_tsv ../task-a-es.tsv
```

## Training RoBERTa Joke Raters

Train language-specific joke rating models (single GPU, no accelerate):

```bash
# English (uses combined_jokes_full.csv)
sbatch jobs/roberta_en.sh

# Chinese (uses zh_data_labeled.csv)
sbatch jobs/roberta_zh.sh

# Spanish (uses es_data_labeled_llama3.1.csv)
sbatch jobs/roberta_es.sh
```

Or run locally:
```bash
python train_roberta.py --language en --output_dir ./checkpoints/roberta-en
python train_roberta.py --language zh --output_dir ./checkpoints/roberta-zh
python train_roberta.py --language es --output_dir ./checkpoints/roberta-es
```

## Training GRPO Joke Generators

Train language-specific joke generation models using GRPO (single GPU, no accelerate):

```bash
# English
sbatch jobs/grpo_en.sh

# Chinese
sbatch jobs/grpo_zh.sh

# Spanish
sbatch jobs/grpo_es.sh
```

Or run locally:
```bash
python train_grpo.py --language en --model_id Qwen/Qwen2.5-7B-Instruct
python train_grpo.py --language zh --model_id Qwen/Qwen2.5-7B-Instruct
python train_grpo.py --language es --model_id Qwen/Qwen2.5-7B-Instruct
```

## Data Files

| Language | RoBERTa Training Data | GRPO Test Data |
|----------|----------------------|----------------|
| English  | `data/combined_jokes_full.csv` | `task-a-en.tsv` |
| Chinese  | `data/zh_data_labeled.csv` | `task-a-zh.tsv` |
| Spanish  | `data/es_data_labeled_llama3.1.csv` | `task-a-es.tsv` |

# Roadmap / TODOs

- increase dataset size for the joke classifier model, especially high rated jokes
- collect news headline datasets and create headline-based jokes
- Set up HPC workflows on BwHPC (8× H200 cluster via SSH)

# Computational Resources
We use the [BwHPC](https://www.bwhpc.de/) infrastructure.
For prototyping we rely on the notebook interface; for final RL training we access 8× H200 GPUs via SSH and the batch queue system.

# Slides

Project slides:
https://docs.google.com/presentation/d/1hKoUTLWRMajmYh5cHBMrzgq0CsKuycSQR-X2dqITbag/edit?slide=id.g3a83e38ea0a_0_50#slide=id.g3a83e38ea0a_0_50
