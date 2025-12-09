"""
Evaluation script for comparing trained joke generation models.

This script:
1. Loads multiple models (base and fine-tuned)
2. Generates jokes for each prompt in the test set
3. Computes various metrics (RoBERTa score, reward functions, diversity)
4. Outputs a comparison CSV and summary statistics
"""

import argparse
import os
import json
from collections import Counter
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from grpo.rewards import (
    roberta_score,
    extract_joke_structure,
    is_valid_single_joke,
    word_pair_prompt_adherence,
    formatting,
    length_penalty,
    coherence_penalty,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare joke generation models")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of model paths or HuggingFace model IDs to evaluate",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional friendly names for models (must match --models length)",
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
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to use (None = all)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=3,
        help="Number of jokes to generate per prompt for diversity metrics",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load a model and tokenizer, handling both base models and LoRA adapters."""
    print(f"Loading model: {model_path}")

    # Check if this is a LoRA adapter (has adapter_config.json)
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        # Load adapter config to get base model
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        print(f"  -> LoRA adapter, base model: {base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Regular model or HuggingFace model ID
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def generate_jokes(
    model,
    tokenizer,
    prompts: list[str],
    num_generations: int = 1,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    batch_size: int = 4,
) -> list[list[str]]:
    """Generate jokes for a list of prompts."""
    all_generations = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Format as chat messages
        batch_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)

        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        batch_generations = [[] for _ in range(len(batch_prompts))]

        for _ in range(num_generations):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the generated tokens
            generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for j, text in enumerate(generated_texts):
                batch_generations[j].append(text.strip())

        all_generations.extend(batch_generations)

    return all_generations


def compute_distinct_n(texts: list[str], n: int = 2) -> float:
    """Compute distinct-n metric (lexical diversity)."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def compute_structure_diversity(jokes: list[str]) -> dict:
    """Compute joke structure distribution."""
    structures = [extract_joke_structure(joke) for joke in jokes]
    counter = Counter(structures)
    total = len(structures)
    return {k: v / total for k, v in counter.items()}


def evaluate_model(
    model,
    tokenizer,
    prompts: list[str],
    model_name: str,
    args,
) -> pd.DataFrame:
    """Evaluate a single model on all prompts."""
    print(f"\nEvaluating: {model_name}")

    # Generate jokes
    all_generations = generate_jokes(
        model,
        tokenizer,
        prompts,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    # Flatten for metrics that need all jokes
    all_jokes = [joke for gens in all_generations for joke in gens]
    all_prompts_expanded = [p for p, gens in zip(prompts, all_generations) for _ in gens]

    # Compute metrics
    print("  Computing RoBERTa scores...")
    roberta_scores = roberta_score(all_jokes)

    print("  Computing reward metrics...")
    word_pair_scores = word_pair_prompt_adherence(all_jokes, all_prompts_expanded)
    formatting_scores = formatting(all_jokes)
    length_scores = length_penalty(all_jokes)
    coherence_scores = coherence_penalty(all_jokes)

    # Validity check
    validity = [is_valid_single_joke(joke) for joke in all_jokes]

    # Build results dataframe
    results = []
    idx = 0
    for prompt_idx, (prompt, gens) in enumerate(zip(prompts, all_generations)):
        for gen_idx, joke in enumerate(gens):
            results.append(
                {
                    "model": model_name,
                    "prompt_idx": prompt_idx,
                    "generation_idx": gen_idx,
                    "prompt": prompt,
                    "joke": joke,
                    "roberta_score": roberta_scores[idx],
                    "word_pair_score": word_pair_scores[idx],
                    "formatting_score": formatting_scores[idx],
                    "length_score": length_scores[idx],
                    "coherence_score": coherence_scores[idx],
                    "is_valid": validity[idx],
                    "word_count": len(joke.split()),
                    "structure": extract_joke_structure(joke),
                }
            )
            idx += 1

    return pd.DataFrame(results)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per model."""
    summary = []

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        all_jokes = model_df["joke"].tolist()

        stats = {
            "model": model_name,
            "n_prompts": model_df["prompt_idx"].nunique(),
            "n_generations": len(model_df),
            # RoBERTa scores
            "roberta_mean": model_df["roberta_score"].mean(),
            "roberta_std": model_df["roberta_score"].std(),
            "roberta_median": model_df["roberta_score"].median(),
            # Validity
            "validity_rate": model_df["is_valid"].mean(),
            # Word pair adherence (exclude None values)
            "word_pair_mean": model_df["word_pair_score"].dropna().mean(),
            # Other rewards
            "formatting_mean": model_df["formatting_score"].mean(),
            "length_mean": model_df["length_score"].mean(),
            "coherence_mean": model_df["coherence_score"].mean(),
            # Diversity metrics
            "distinct_1": compute_distinct_n(all_jokes, 1),
            "distinct_2": compute_distinct_n(all_jokes, 2),
            "avg_word_count": model_df["word_count"].mean(),
            # Structure diversity (entropy)
            "n_unique_structures": model_df["structure"].nunique(),
        }

        # Add structure distribution
        structure_dist = compute_structure_diversity(all_jokes)
        for struct, ratio in structure_dist.items():
            stats[f"structure_{struct}"] = ratio

        summary.append(stats)

    return pd.DataFrame(summary)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_df = pd.read_parquet(args.test_data)
    prompts = test_df["prompt"].tolist()

    if args.num_samples:
        prompts = prompts[: args.num_samples]
        print(f"Using {len(prompts)} samples")

    # Set model names
    model_names = args.model_names if args.model_names else args.models
    if len(model_names) != len(args.models):
        print("Warning: model_names length doesn't match models, using model paths as names")
        model_names = args.models

    # Evaluate each model
    all_results = []
    for model_path, model_name in zip(args.models, model_names):
        model, tokenizer = load_model_and_tokenizer(model_path)
        results_df = evaluate_model(model, tokenizer, prompts, model_name, args)
        all_results.append(results_df)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Compute summary statistics
    summary_df = compute_summary_stats(combined_df)

    # Save results
    detailed_path = os.path.join(args.output_dir, f"eval_detailed_{timestamp}.csv")
    summary_path = os.path.join(args.output_dir, f"eval_summary_{timestamp}.csv")

    combined_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Print summary table
    display_cols = [
        "model",
        "roberta_mean",
        "roberta_std",
        "validity_rate",
        "word_pair_mean",
        "distinct_2",
        "avg_word_count",
    ]
    print(summary_df[display_cols].to_string(index=False))

    print(f"\nDetailed results saved to: {detailed_path}")
    print(f"Summary results saved to: {summary_path}")

    # Print example generations for comparison
    print(f"\n{'='*60}")
    print("EXAMPLE GENERATIONS (first prompt)")
    print("=" * 60)
    first_prompt = prompts[0]
    print(f"Prompt: {first_prompt}\n")

    for model_name in model_names:
        model_examples = combined_df[
            (combined_df["model"] == model_name) & (combined_df["prompt_idx"] == 0)
        ]
        print(f"--- {model_name} ---")
        for _, row in model_examples.iterrows():
            print(f"  [{row['roberta_score']:.2f}] {row['joke']}")
        print()


if __name__ == "__main__":
    main()
