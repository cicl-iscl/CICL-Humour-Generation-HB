import os
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
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

    # For FSDP/DDP: don't set device_map, let Accelerate handle placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    model_name = args.model_id.split("/")[-1] + "-Jokester"
    # Configure GRPO Training
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
        save_strategy="steps",
        save_steps=500,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        push_to_hub=True,
        hub_model_id=f"KonradBRG/{model_name}",
        # For distributed training
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    print("Starting GRPO training...")
    trainer.train()
    print("Training complete.")

    # Save the final model
    trainer.save_model()
    print(f"Model saved to {args.output_dir}")

    # Only run generation demo on main process
    if trainer.accelerator.is_main_process:
        print("\n--- Example Joke Generation Post-Training ---")
        model.eval()

        prompt = "Generate the funniest possible joke that contains these two words: 'microwave', 'shoes'. Return only the joke."
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
                response = response[len(prompt):].strip()
                print(f"Joke {i+1}: {response}")
                print("-" * 24)


if __name__ == "__main__":
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
    main()