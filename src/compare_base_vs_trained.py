"""
Compare mean reward scores of base Qwen2.5-7B-Instruct vs GRPO-trained models
on the test set for each language (en, zh, es).

Uses the same composite reward used during GRPO training (not just argmax class),
including expected-value RoBERTa scoring for finer granularity.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from grpo.rewards import (
    create_is_valid_single_joke_fn,
    create_formatting_fn,
    create_length_penalty_fn,
    create_coherence_penalty_fn,
    word_pair_prompt_adherence,
)


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
    )
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=32,
        help="Batch size for reward model scoring",
    )
    return parser.parse_args()


def load_generative_model(model_id: str):
    """Load a causal LM and tokenizer."""
    print(f"Loading generative model: {model_id}")
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


def load_reward_model(model_id: str, device: str = "cuda"):
    """Load the RoBERTa reward model directly (not via pipeline)."""
    print(f"Loading reward model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = model.to(device).eval()
    return model, tokenizer


def score_expected_value(
    reward_model,
    reward_tokenizer,
    texts: list[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> list[float]:
    """
    Score texts using expected value E[score] = sum(p_i * i) for i in 0..10.

    This gives a continuous score rather than the coarse argmax class,
    making differences between base and trained models visible.
    """
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = reward_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = reward_model(**inputs)
            logits = outputs["logits"]  # [batch, 11] for classes 0-10
            probs = F.softmax(logits, dim=-1)
            # Expected value: sum(p_i * i) for i in 0..10
            class_indices = torch.arange(
                logits.size(-1), device=device, dtype=torch.float
            )
            expected = (probs * class_indices).sum(dim=-1)
            all_scores.extend(expected.cpu().tolist())

    return all_scores


def generate_jokes(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
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


def compute_composite_reward(
    jokes: list[str],
    prompts: list[str],
    ev_scores: list[float],
    lang: str,
) -> list[float]:
    """
    Compute the same composite reward used during GRPO training.

    Weights match train_grpo.py:
      roberta: 1.0, word_pair: 2.0, formatting: 0.5, length: 0.5, coherence: 0.5
    (structure_diversity and headline_adherence omitted as they depend on
    generation order / are minor)
    """
    validator = create_is_valid_single_joke_fn(lang)
    formatting_fn = create_formatting_fn(lang)
    length_fn = create_length_penalty_fn(lang)
    coherence_fn = create_coherence_penalty_fn(lang)

    # Compute sub-rewards
    fmt_scores = formatting_fn(jokes)
    len_scores = length_fn(jokes)
    coh_scores = coherence_fn(jokes)
    wp_scores = word_pair_prompt_adherence(jokes, prompts)

    composite = []
    for i in range(len(jokes)):
        # RoBERTa EV score (gated by validity like in training)
        if validator(jokes[i]):
            r = ev_scores[i]
        else:
            r = 0.0

        wp = wp_scores[i] if wp_scores[i] is not None else 0.0

        total = (
            1.0 * r
            + 2.0 * wp
            + 0.5 * fmt_scores[i]
            + 0.5 * len_scores[i]
            + 0.5 * coh_scores[i]
        )
        composite.append(total)

    return composite


def evaluate_language(lang: str, args):
    """Run base vs trained comparison for one language."""
    config = LANG_CONFIG[lang]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test prompts
    test_path = os.path.join(args.data_dir, config["test_data"])
    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}")
    print(f"{'='*60}")
    test_df = pd.read_parquet(test_path)
    prompts = test_df["prompt"].tolist()
    if args.num_samples:
        prompts = prompts[: args.num_samples]
    print(f"Using {len(prompts)} prompts")

    # Load reward model once for this language
    reward_model, reward_tokenizer = load_reward_model(config["reward_model"], device)
    validator = create_is_valid_single_joke_fn(lang)

    results = {}
    for label, model_id in [("base", BASE_MODEL), ("trained", config["trained_model"])]:
        print(f"\n--- {label}: {model_id} ---")
        gen_model, gen_tokenizer = load_generative_model(model_id)
        jokes = generate_jokes(
            gen_model,
            gen_tokenizer,
            prompts,
            args.max_new_tokens,
            args.temperature,
            args.batch_size,
        )

        # Free generative model before scoring
        del gen_model
        torch.cuda.empty_cache()

        # Score with expected value
        print("Scoring with reward model (expected value)...")
        ev_scores = score_expected_value(
            reward_model, reward_tokenizer, jokes, args.reward_batch_size, device
        )

        # Compute composite reward (mirrors GRPO training)
        composite_scores = compute_composite_reward(jokes, prompts, ev_scores, lang)

        # Validity
        validity = [validator(j) for j in jokes]

        # Argmax scores for comparison
        argmax_scores = []
        for s, v in zip(ev_scores, validity):
            argmax_scores.append(s if v else 0.0)

        results[label] = {
            "model_id": model_id,
            "mean_ev_reward": np.mean(ev_scores),
            "mean_ev_reward_valid_only": np.mean(
                [s for s, v in zip(ev_scores, validity) if v]
            ) if any(validity) else 0.0,
            "mean_composite_reward": np.mean(composite_scores),
            "std_composite_reward": np.std(composite_scores),
            "validity_rate": np.mean(validity),
            "num_samples": len(prompts),
            "jokes": jokes,
            "ev_scores": ev_scores,
            "composite_scores": composite_scores,
        }

        print(f"  EV reward (all):        {results[label]['mean_ev_reward']:.4f}")
        print(f"  EV reward (valid only):  {results[label]['mean_ev_reward_valid_only']:.4f}")
        print(f"  Composite reward:        {results[label]['mean_composite_reward']:.4f}")
        print(f"  Validity rate:           {results[label]['validity_rate']:.2%}")

    # Free reward model
    del reward_model
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
                "mean_ev_reward": r["mean_ev_reward"],
                "mean_ev_reward_valid_only": r["mean_ev_reward_valid_only"],
                "mean_composite_reward": r["mean_composite_reward"],
                "std_composite_reward": r["std_composite_reward"],
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
            for joke, ev, comp in zip(
                r["jokes"], r["ev_scores"], r["composite_scores"]
            ):
                detail_rows.append({
                    "language": lang,
                    "model": label,
                    "joke": joke,
                    "ev_reward": ev,
                    "composite_reward": comp,
                })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(args.output_dir, f"comparison_detailed_{timestamp}.csv")
    detail_df.to_csv(detail_path, index=False)

    # Print final summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print("=" * 60)
    display_cols = [
        "language", "model", "mean_ev_reward", "mean_composite_reward",
        "validity_rate",
    ]
    print(summary_df[display_cols].to_string(index=False))

    print(f"\n{'='*60}")
    print("IMPROVEMENT (trained - base)")
    print("=" * 60)
    for lang in args.languages:
        b = all_results[lang]["base"]
        t = all_results[lang]["trained"]
        d_ev = t["mean_ev_reward"] - b["mean_ev_reward"]
        d_comp = t["mean_composite_reward"] - b["mean_composite_reward"]
        d_val = t["validity_rate"] - b["validity_rate"]
        print(f"  {lang.upper()}:")
        print(f"    EV reward:        {b['mean_ev_reward']:.4f} -> {t['mean_ev_reward']:.4f}  ({d_ev:+.4f})")
        print(f"    Composite reward: {b['mean_composite_reward']:.4f} -> {t['mean_composite_reward']:.4f}  ({d_comp:+.4f})")
        print(f"    Validity rate:    {b['validity_rate']:.2%} -> {t['validity_rate']:.2%}  ({d_val:+.2%})")

    print(f"\nSummary: {summary_path}")
    print(f"Details: {detail_path}")


if __name__ == "__main__":
    main()
