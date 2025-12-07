import argparse
import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from grpo.rewards import (
    roberta_score,
    structure_diversity_reward,
    word_pair_prompt_adherence,
    formatting,
    length_penalty,
    headline_adherence,
    coherence_penalty
)
from grpo.cli import parse_args


def main():
    """Main function to set up and run GRPO training for joke generation."""
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
        use_vllm=False,
        vllm_mode=None,
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