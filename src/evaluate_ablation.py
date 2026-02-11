"""
Ablation evaluation: compare full GRPO model vs 3 ablation models.

Loads 4 models (full + no-formatting + no-classifier + no-diversity),
generates jokes on the EN test set, and reports:
  - RoBERTa EV score (funniness)
  - Distinct-2 (lexical diversity)
  - Structure entropy (structural diversity)
  - Validity rate
"""

import argparse
import math
import os
from collections import Counter
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from evaluate_models import (
    compute_distinct_n,
    generate_jokes,
    get_structure_extractor,
    load_model_and_tokenizer,
    load_reward_model,
    score_expected_value,
)
from grpo.rewards import create_is_valid_single_joke_fn

ABLATION_MODELS = {
    "full": "./checkpoints/grpo-en",
    "no-formatting": "./checkpoints/grpo-en-no-formatting",
    "no-classifier": "./checkpoints/grpo-en-no-classifier",
    "no-diversity": "./checkpoints/grpo-en-no-diversity",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO reward ablation models"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/rl_df_test.parquet",
        help="Path to test data parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test prompts (None = all)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=3,
        help="Jokes per prompt for diversity metrics",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=32,
        help="Batch size for reward model scoring",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh", "es"],
        help="Language for evaluation",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Override model paths as name:path pairs (e.g., full:./checkpoints/grpo-en)",
    )
    return parser.parse_args()


def compute_structure_entropy(jokes: list[str], language: str = "en") -> float:
    """Compute Shannon entropy of joke structure distribution."""
    extractor = get_structure_extractor(language)
    structures = [extractor(joke) for joke in jokes]
    counter = Counter(structures)
    total = len(structures)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def evaluate_ablation_model(
    model_name: str,
    model_path: str,
    prompts: list[str],
    reward_model,
    reward_tokenizer,
    args,
) -> dict:
    """Evaluate a single ablation model and return summary metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    language = args.language
    is_valid = create_is_valid_single_joke_fn(language)

    model, tokenizer = load_model_and_tokenizer(model_path)

    all_generations = generate_jokes(
        model,
        tokenizer,
        prompts,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    all_jokes = [joke for gens in all_generations for joke in gens]

    # RoBERTa EV score
    roberta_scores = score_expected_value(
        reward_model, reward_tokenizer, all_jokes, args.reward_batch_size, device
    )

    # Validity
    validity = [is_valid(joke) for joke in all_jokes]

    return {
        "model": model_name,
        "model_path": model_path,
        "n_prompts": len(prompts),
        "n_jokes": len(all_jokes),
        "roberta_ev_mean": sum(roberta_scores) / len(roberta_scores),
        "roberta_ev_std": pd.Series(roberta_scores).std(),
        "distinct_2": compute_distinct_n(all_jokes, 2),
        "structure_entropy": compute_structure_entropy(all_jokes, language),
        "validity_rate": sum(validity) / len(validity),
    }


REWARD_MODELS = {
    "en": "KonradBRG/joke-rater-roberta-en",
    "zh": "KonradBRG/joke-rater-roberta-zh",
    "es": "KonradBRG/joke-rater-roberta-es",
}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve model dict
    models = dict(ABLATION_MODELS)
    if args.models:
        models = {}
        for pair in args.models:
            name, path = pair.split(":", 1)
            models[name] = path

    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_df = pd.read_parquet(args.test_data)
    prompts = test_df["prompt"].tolist()
    if args.num_samples:
        prompts = prompts[: args.num_samples]
    print(f"Evaluating on {len(prompts)} prompts")

    # Load reward model once
    reward_model_id = REWARD_MODELS[args.language]
    reward_model, reward_tokenizer = load_reward_model(reward_model_id, device)

    # Evaluate each model
    results = []
    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({path})")
        print("=" * 60)
        metrics = evaluate_ablation_model(
            name, path, prompts, reward_model, reward_tokenizer, args
        )
        results.append(metrics)

    del reward_model
    torch.cuda.empty_cache()

    # Build summary table
    summary_df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"ablation_summary_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False)

    # Print formatted comparison
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    display_cols = [
        "model",
        "roberta_ev_mean",
        "roberta_ev_std",
        "distinct_2",
        "structure_entropy",
        "validity_rate",
    ]
    print(summary_df[display_cols].to_string(index=False))

    # Print delta from full model
    if "full" in summary_df["model"].values:
        print(f"\n{'='*80}")
        print("DELTA FROM FULL MODEL")
        print("=" * 80)
        full_row = summary_df[summary_df["model"] == "full"].iloc[0]
        for _, row in summary_df.iterrows():
            if row["model"] == "full":
                continue
            print(f"\n  {row['model']}:")
            print(
                f"    roberta_ev:        {row['roberta_ev_mean'] - full_row['roberta_ev_mean']:+.4f}"
            )
            print(
                f"    distinct_2:        {row['distinct_2'] - full_row['distinct_2']:+.4f}"
            )
            print(
                f"    structure_entropy: {row['structure_entropy'] - full_row['structure_entropy']:+.4f}"
            )
            print(
                f"    validity_rate:     {row['validity_rate'] - full_row['validity_rate']:+.4f}"
            )

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
