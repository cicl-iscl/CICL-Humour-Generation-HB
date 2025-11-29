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
└── humor-archi.drawio  # System architecture diagram
```

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
