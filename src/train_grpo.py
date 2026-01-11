"""
Train a language-specific GRPO joke generation model.

Usage (single GPU, no accelerate):
    python train_grpo.py --language en --model_id Qwen/Qwen2.5-7B-Instruct
    python train_grpo.py --language zh --model_id Qwen/Qwen2.5-7B-Instruct
    python train_grpo.py --language es --model_id Qwen/Qwen2.5-7B-Instruct
"""
import os
import torch
import weave
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from dotenv import load_dotenv
from grpo.rewards import (
    create_roberta_score_fn,
    create_structure_diversity_reward_fn,
    create_formatting_fn,
    create_length_penalty_fn,
    create_headline_adherence_fn,
    create_coherence_penalty_fn,
    word_pair_prompt_adherence,
)
from grpo.cli import parse_args

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GenerationLoggingCallback(TrainerCallback):
    """Callback to log sample generations with reward metrics to wandb."""

    def __init__(
        self,
        tokenizer,
        reward_fns,
        reward_names,
        reward_weights,
        sample_prompts,
        log_every_n_steps: int = 50,
        num_samples: int = 4,
    ):
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.reward_names = reward_names
        self.reward_weights = reward_weights
        self.sample_prompts = sample_prompts
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        if state.global_step % self.log_every_n_steps != 0:
            return

        if model is None:
            return

        # Generate samples
        model.eval()
        prompts = self.sample_prompts[: self.num_samples]
        generations = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the generated part
                generation = response[len(prompt):].strip()
                generations.append(generation)

        model.train()

        # Compute rewards for each generation
        table_data = []
        for i, (prompt, generation) in enumerate(zip(prompts, generations)):
            row = {
                "step": state.global_step,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "generation": generation,
            }

            total_weighted_reward = 0.0
            for fn, name, weight in zip(self.reward_fns, self.reward_names, self.reward_weights):
                try:
                    # Call reward function with single completion
                    if "prompts" in fn.__code__.co_varnames:
                        scores = fn([generation], prompts=[prompt])
                    else:
                        scores = fn([generation])
                    score = scores[0] if scores[0] is not None else 0.0
                except Exception:
                    score = 0.0

                row[name] = round(score, 3) if score is not None else None
                if score is not None:
                    total_weighted_reward += weight * score

            row["total_reward"] = round(total_weighted_reward, 3)
            table_data.append(row)

        # Log to wandb as a table
        columns = ["step", "prompt", "generation"] + self.reward_names + ["total_reward"]
        table = wandb.Table(columns=columns)
        for row in table_data:
            table.add_data(*[row.get(col, None) for col in columns])

        wandb.log({f"generations/step_{state.global_step}": table}, step=state.global_step)


def main():
    """Main function to set up and run GRPO training for joke generation."""
    args = parse_args()

    # Load environment variables (e.g., WANDB_API_KEY, HF_TOKEN)
    load_dotenv()

    # Load datasets
    print(f"Loading datasets from {args.train_data_file} and {args.test_data_file}...")
    print(f"Training language: {args.language}")
    try:
        train_dataset = load_dataset(
            "parquet", data_files=args.train_data_file, split="train"
        )
        test_dataset = load_dataset(
            "parquet", data_files=args.test_data_file, split="train"
        )
    except Exception as e:
        print(f"Error loading datasets. Ensure the paths are correct. Error: {e}")
        return

    # Single GPU training - explicitly set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Ensure model is in training mode with gradients enabled
    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create language-aware reward functions
    lang = args.language
    roberta_score_fn = create_roberta_score_fn(args.joke_rater_model, language=lang)
    structure_diversity_fn = create_structure_diversity_reward_fn(lang)
    formatting_fn = create_formatting_fn(lang)
    length_penalty_fn = create_length_penalty_fn(lang)
    headline_adherence_fn = create_headline_adherence_fn(lang)
    coherence_penalty_fn = create_coherence_penalty_fn(lang)

    print(f"Using joke rater model: {args.joke_rater_model}")
    print(f"All reward functions configured for language: {lang}")

    # Define Reward Function List, Names, and Weights
    reward_fns = [
        roberta_score_fn,
        structure_diversity_fn,
        word_pair_prompt_adherence,
        formatting_fn,
        length_penalty_fn,
        headline_adherence_fn,
        coherence_penalty_fn,
    ]
    reward_names = [
        "roberta_score",
        "structure_diversity",
        "word_pair_adherence",
        "formatting",
        "length_penalty",
        "headline_adherence",
        "coherence_penalty",
    ]
    reward_weights = [1.0, 1.5, 2.0, 0.5, 0.5, 2.0, 0.5]

    lang_suffix = {"en": "English", "zh": "Chinese", "es": "Spanish"}[args.language]
    model_name = f"{args.model_id.split('/')[-1]}-Jokester-{lang_suffix}"

    # Get sample prompts for logging callback
    sample_prompts = train_dataset["prompt"][:8]  # Take first 8 prompts for sampling

    # Configure GRPO Training
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        report_to=args.report_to,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        reward_weights=reward_weights,
        learning_rate=args.learning_rate,
        eval_strategy="epoch" if args.max_steps <= 0 else "no",
        save_strategy="no",
        save_steps=500 if args.max_steps <= 0 else min(50, args.max_steps),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        push_to_hub=True,
        hub_model_id=f"KonradBRG/{model_name}",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Create generation logging callback
    generation_callback = GenerationLoggingCallback(
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        reward_names=reward_names,
        reward_weights=reward_weights,
        sample_prompts=sample_prompts,
        log_every_n_steps=50,
        num_samples=4,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[generation_callback],
    )

    print(f"Starting GRPO training for {lang_suffix} jokes...")
    trainer.train()
    print("Training complete.")

    # Save the final model
    trainer.save_model()
    print(f"Model saved to {args.output_dir}")

    # Run generation demo
    print(f"\n--- Example {lang_suffix} Joke Generation Post-Training ---")
    model.eval()

    # Language-specific prompts
    prompts = {
        "en": "Generate the funniest possible joke that contains these two words: 'microwave', 'shoes'. Return only the joke.",
        "zh": "生成一个包含这两个词的最有趣的笑话：'电脑', '咖啡'。只返回笑话。",
        "es": "Genera el chiste más gracioso posible que contenga estas dos palabras: 'computadora', 'café'. Devuelve solo el chiste.",
    }
    prompt = prompts[args.language]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        for i in range(5):
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt) :].strip()
            print(f"Joke {i+1}: {response}")
            print("-" * 24)


if __name__ == "__main__":
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    main()
