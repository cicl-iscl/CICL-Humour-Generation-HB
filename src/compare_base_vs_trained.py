"""
Compare mean reward scores of base Qwen2.5-7B-Instruct vs GRPO-trained models
on the test set for each language (en, zh, es).

For each language:
1. Load test prompts from the parquet test set
2. Generate jokes with base model and trained model
3. Score all jokes with the language-specific RoBERTa reward model
4. Report mean reward comparison
"""

import argparse
import os
import json
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.rewards import create_roberta_score_fn, create_is_valid_single_joke_fn


LANG_CONFIG = {
    "en": {
        "test_data": "rl_df_test.parquet",
        "trained_model": "KonradBRG/Qwen2.5-7B-Instruct-Jokester-English",
        "reward_model": "KonradBRG/joke-rater-roberta-en",
    },
    "zh": {
        "test_data": "rl_df_test_zh.parquet",
        "trained_model": "KonradBRG/Qwen2.5-7B-Instruct-Jokester-Chinese",
        "reward_model": "KonradBRG/joke-rater-roberta-zh",
    },
    "es": {
        "test_data": "rl_df_test_es.parquet",
        "trained_model": "KonradBRG/Qwen2.5-7B-Instruct-Jokester-Spanish",
        "reward_model": "KonradBRG/joke-rater-roberta-es",
    },
}

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare base vs trained model reward scores"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en", "zh", "es"],
        choices=["en", "zh", "es"],
        help="Languages to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Directory containing test parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to use (None = all)",
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


def load_model(model_id: str):
    """Load a causal LM and tokenizer."""
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
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
    model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, batch_size: int
) -> list[str]:
    """Generate one joke per prompt."""
    device = next(model.parameters()).device
    all_jokes = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        batch_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        generated_texts = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        all_jokes.extend([t.strip() for t in generated_texts])

    return all_jokes


def evaluate_language(lang: str, args):
    """Run base vs trained comparison for one language. Returns dict of results."""
    config = LANG_CONFIG[lang]

    # Load test prompts
    test_path = os.path.join(args.data_dir, config["test_data"])
    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}")
    print(f"Loading test data: {test_path}")
    test_df = pd.read_parquet(test_path)
    prompts = test_df["prompt"].tolist()
    if args.num_samples:
        prompts = prompts[: args.num_samples]
    print(f"Using {len(prompts)} prompts")

    # Create reward scorer
    reward_fn = create_roberta_score_fn(config["reward_model"], language=lang)
    validator = create_is_valid_single_joke_fn(lang)

    results = {}
    for label, model_id in [("base", BASE_MODEL), ("trained", config["trained_model"])]:
        print(f"\n--- {label} model: {model_id} ---")
        model, tokenizer = load_model(model_id)
        jokes = generate_jokes(
            model, tokenizer, prompts,
            args.max_new_tokens, args.temperature, args.batch_size,
        )

        # Score
        print("Scoring with reward model...")
        scores = reward_fn(jokes)
        validity = [validator(j) for j in jokes]

        results[label] = {
            "model_id": model_id,
            "mean_reward": sum(scores) / len(scores),
            "median_reward": sorted(scores)[len(scores) // 2],
            "std_reward": pd.Series(scores).std(),
            "validity_rate": sum(validity) / len(validity),
            "num_samples": len(prompts),
            "jokes": jokes,
            "scores": scores,
        }

        print(f"  Mean reward: {results[label]['mean_reward']:.4f}")
        print(f"  Validity:    {results[label]['validity_rate']:.2%}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}
    summary_rows = []

    for lang in args.languages:
        lang_results = evaluate_language(lang, args)
        all_results[lang] = lang_results

        for label in ["base", "trained"]:
            r = lang_results[label]
            summary_rows.append({
                "language": lang,
                "model": label,
                "model_id": r["model_id"],
                "mean_reward": r["mean_reward"],
                "median_reward": r["median_reward"],
                "std_reward": r["std_reward"],
                "validity_rate": r["validity_rate"],
                "num_samples": r["num_samples"],
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, f"comparison_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    # Save detailed per-joke results
    detail_rows = []
    for lang, lang_results in all_results.items():
        for label in ["base", "trained"]:
            r = lang_results[label]
            for joke, score in zip(r["jokes"], r["scores"]):
                detail_rows.append({
                    "language": lang,
                    "model": label,
                    "joke": joke,
                    "reward": score,
                })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(args.output_dir, f"comparison_detailed_{timestamp}.csv")
    detail_df.to_csv(detail_path, index=False)

    # Print final summary table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Print improvement
    print(f"\n{'='*60}")
    print("IMPROVEMENT (trained - base)")
    print("=" * 60)
    for lang in args.languages:
        base_mean = all_results[lang]["base"]["mean_reward"]
        trained_mean = all_results[lang]["trained"]["mean_reward"]
        delta = trained_mean - base_mean
        print(f"  {lang.upper()}: {base_mean:.4f} -> {trained_mean:.4f}  (delta: {delta:+.4f})")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Detailed results saved to: {detail_path}")


if __name__ == "__main__":
    main()
