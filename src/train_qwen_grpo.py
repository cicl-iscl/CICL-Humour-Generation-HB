import argparse
import os
import re
import emoji
import torch
import numpy as np
from collections import deque
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


def is_valid_single_joke(text):
    """
    Combat joke stacking.
    """
    if text.count("?") > 1:
        return False
    # Using 'why ' instead of 'why ' is crucial due to tokenization/word boundaries
    if text.lower().count("why ") > 1:
        return False
    if "Q:" in text.lower() or "A:" in text.lower():
        return False
    if '\"' in text or '-' in text: # This is a strong heuristic; consider refining it if it's too aggressive.
        return False
    if len(text.strip().split('\n')) > 3:
        return False
    
    return True

recent_structures = deque(maxlen=30) 

def extract_joke_structure(joke: str) -> str:
    """Extracts common joke structures using regex."""
    joke_lower = joke.lower()
    
    if re.search(r"why\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower):
        return "why-did"
    elif re.search(r"where\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower):
        return "where-did"
    elif re.search(r"how\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower):
        return "how-did"
    elif re.search(r"what\s+do\s+you\s+call", joke_lower):
        return "what-do-you-call"
    elif re.search(r"knock\s+knock", joke_lower):
        return "knock-knock"
    elif joke.count("?") == 1 and (joke.count("!") >= 1 or joke.count(".") >= 1):
        return "qa-punchline"
    elif any(phrase in joke_lower for phrase in [" is like ", " is when "]):
        return "observation"
    else:
        return "one-liner"

def structure_diversity_reward(completions, **kwargs):
    """Calculates reward based on structure novelty relative to recent history (cell 6)."""
    global recent_structures
    scores = []
    freq = {}
    
    # Calculate frequencies of existing structures in the history
    for s in recent_structures:
        freq[s] = freq.get(s, 0) + 1

    total = max(len(recent_structures), 1)
    num_structures = len(freq) if freq else 1
    # Target is equal distribution (e.g., if 5 structures, target is 1/5 = 0.2)
    target = 1 / num_structures 

    for joke in completions:
        s = extract_joke_structure(joke)
        actual = freq.get(s, 0) / total
        reward = target - actual
        scores.append(reward)
        
        # Update history and frequency for the NEXT joke/step
        recent_structures.append(s)
        freq[s] = freq.get(s, 0) + 1
        total += 1
    
    return scores

scoring_pipe = None

def roberta_score(completions, **kwargs):
    """
    Scores humor using RoBERTA, but only for valid outputs (cell 7).
    Invalid outputs get 0.0 regardless of what the model thinks.
    """
    global scoring_pipe
    if scoring_pipe is None:
        # Load the pipeline on the first call
        print("Loading RoBERTA Joke Rater Pipeline...")
        scoring_pipe = pipeline(
            "text-classification", 
            model="KonradBRG/RoBERTA-Joke-Rater", 
            trust_remote_code=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("RoBERTA Pipeline loaded.")

    scores = []
    # Only score completions that are valid jokes to reduce computation
    valid_completions = [c for c in completions if is_valid_single_joke(c)]
    valid_indices = [i for i, c in enumerate(completions) if is_valid_single_joke(c)]
    
    # Initialize all scores to 0.0 (for invalid jokes)
    final_scores = [0.0] * len(completions)
    
    if valid_completions:
        # Get scores for valid jokes
        roberta_labels = scoring_pipe(valid_completions)
        for idx, roberta_label in zip(valid_indices, roberta_labels):
            roberta_score = float(roberta_label["label"])
            final_scores[idx] = roberta_score
            
    return final_scores


def word_pair_prompt_adherence(completions, prompts, **kwargs):
    """Enforces the word pair constraint (cell 8)."""
    scores = []
    pattern = r"contains\s+these\s+two\s+words:\s*'([^']+)'\s*,\s*'([^']+)'"
    for i in range(len(completions)):
        p = prompts[i]
        # Skip if not a word pair task
        if "two words" not in p:
            scores.append(None)
            continue
        
        c = completions[i].lower()
        try:
            # Extract the two required words from the prompt
            w1, w2 = re.findall(pattern, p, flags=re.IGNORECASE)[0]
        except IndexError:
            scores.append(None)
            continue
        
        w1_lower = w1.lower().strip()
        w2_lower = w2.lower().strip()
        
        # Check if words are found in the completion
        w1_found = w1_lower in c
        w2_found = w2_lower in c
        
        # Reward/Penalty logic:
        if w1_found and w2_found:
            scores.append(2.0) # High positive reward for both
        elif w1_found or w2_found:
            scores.append(-1.0) # Penalty for only one
        else:
            scores.append(-2.0) # High penalty for neither
    
    return scores

def headline_adherence(completions, prompts, **kwargs):
    """Simple check for headline tasks (cell 8)."""
    scores = []
    for i, completion in enumerate(completions):
        
        # Skip if it's a word pair task
        if "two words" in prompts[i]: 
            scores.append(None)
            continue
        
        # The prompt is a headline task.
        if len(completion.split()) <= 25: 
            # Check for bad patterns (like repeating the prompt or being conversational)
            if "headline" in completion.lower() or "generate" in completion.lower():
                scores.append(-1.0)
            else:
                scores.append(1.0) # Small positive reward for conforming to length/format
        else:
            scores.append(-1.0)
    return scores

def contains_emoji_func(text):
    """Helper to detect emojis."""
    return any(char in emoji.EMOJI_DATA for char in text)

def formatting(completions, **kwargs):
    """Validates output formatting and penalizes hacking patterns (cell 9)."""
    scores = []
    for completion in completions:
        is_penalized = False
        # Penalties for conversational artifacts/bad symbols
        if ("#" in completion
            or "How about: " in completion
            or "This joke" in completion
            or "Let me know" in completion
            or "Note: " in completion
            or "   " in completion # Multiple spaces
            or contains_emoji_func(completion)
        ):
            scores.append(-1.0)
            is_penalized = True
        # Penalty for joke stacking/invalid structure
        elif not is_valid_single_joke(completion):
            scores.append(-1.0)
            is_penalized = True
        
        if not is_penalized:
            scores.append(1.0)
    
    return scores

def length_penalty(completions, **kwargs):
    """Penalizes outputs outside the optimal 5-24 word range (cell 9)."""
    scores = []
    optimal_length = 16
    max_allowed = 24
    min_allowed = 5
    
    for completion in completions:
        word_count = len(completion.split())
        
        if word_count > max_allowed or word_count < min_allowed:
            scores.append(-2.0) # Severe penalty
        else:
            # Smooth penalty for being over optimal length (16 words)
            deviation = max(0, word_count - optimal_length)
            penalty = -0.2 * deviation
            scores.append(penalty)
    
    return scores

def compute_coherence_penalty(joke: str, penalty_weight: float = 0.5) -> float:
    """Penalize incoherent jokes with rare/technical terms (cell 9)."""
    rare_word_pattern = r'\b[A-Z][a-z]*(?:-[a-z]+)*\b'
    rare_words = len(re.findall(rare_word_pattern, joke))
    
    words = joke.split()
    if len(words) > 0:
        rare_word_ratio = rare_words / len(words)
        # Penalize if capitalized words make up more than 20% of the joke
        if rare_word_ratio > 0.2: 
            return -penalty_weight * (rare_word_ratio - 0.2)
    
    return 0.0

def coherence_penalty(completions, **kwargs):
    """Wrapper for the coherence penalty """
    return [compute_coherence_penalty(c) for c in completions]


def parse_args():
    """Parses command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a Joke Generation Policy using GRPO.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID for the policy model (LLM) to be trained."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Llama-3.2-1B-Instruct-GRPO",
        help="Directory to save the trained GRPO model and logs."
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="data/rl_df_train.parquet",
        help="Path to the training dataset (.parquet file)."
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default="data/rl_df_test.parquet",
        help="Path to the evaluation dataset (.parquet file)."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for the GRPO optimization."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Whether to use vLLM for efficient generation."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Reporting tool to use (e.g., 'wandb', 'none')."
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=64,
        help="Maximum length of the generated joke completions."
    )
    
    # Batch/Generation settings
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per device for evaluation.")
    parser.add_argument("--generation_batch_size", type=int, default=16, help="Batch size for parallel generation.")
    parser.add_argument("--num_generations", type=int, default=16, help="Number of responses to generate per prompt (G in GRPO).")

    return parser.parse_args()


def main():
    
    args = parse_args()
    
    # Load environment variables (e.g., WANDB_API_KEY, HF_TOKEN)
    load_dotenv()
    
    # Load datasets
    print(f"Loading datasets from {args.train_data_file} and {args.test_data_file}...")
    try:
        train_dataset = load_dataset("parquet", data_files=args.train_data_file, split="train")
        test_dataset = load_dataset("parquet", data_files=args.test_data_file, split="train")
    except Exception as e:
        print(f"Error loading datasets. Ensure the paths are correct. Error: {e}")
        return

    # Define Reward Function List and Weights (matching cell 10)
    reward_fns = [
        roberta_score,
        structure_diversity_reward,
        word_pair_prompt_adherence,
        formatting,
        length_penalty,
        headline_adherence,
        coherence_penalty
    ]
    reward_weights = [1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 1.0]

    # Configure GRPO Training (matching cell 10)
    training_args = GRPOConfig(
        output_dir=args.output_dir, 
        report_to=args.report_to,
        num_train_epochs=args.num_train_epochs,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        max_completion_length=args.max_completion_length,
        temperature=0.5,
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        reward_weights=reward_weights,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    print("Starting GRPO training...")
    try:
        import wandb
        wandb.init(project="huggingface", config=args)
    except Exception as e:
        print(f"Could not initialize wandb/weave. Training will continue without detailed logging. Error: {e}")
        
    trainer.train()
    print("Training complete.")
    print("\n--- Example Joke Generation Post-Training ---")
    model = trainer.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    prompt = "Generate the funniest possible joke that contains these two words: 'microwave', 'shoes'. Return only the joke."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    for i in range(5):
        outputs = model.generate(
            **inputs,
            max_length=50,
            temperature=0.8,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        print(f"Joke {i+1}: {response}")
        print("-" * 24)


if __name__ == "__main__":
    # Suppress vLLM logging via environment variable (matching cell 1)
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0' 
    main()