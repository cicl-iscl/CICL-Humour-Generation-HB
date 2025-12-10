import os
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
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
    """
    Main function to set up and run GRPO training for joke generation on 32B.

    This script uses FSDP (via Accelerate) with LoRA for memory-efficient training.
    For 32B model on 4x A100 (320GB VRAM), we use:
    - Full precision BF16 model sharded via FSDP
    - LoRA adapters for parameter-efficient fine-tuning
    - Gradient checkpointing for memory savings
    """
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Load datasets
    print(f"Loading datasets from {args.train_data_file} and {args.test_data_file}...")
    try:
        train_dataset = load_dataset("parquet", data_files=args.train_data_file, split="train")
        test_dataset = load_dataset("parquet", data_files=args.test_data_file, split="train")
    except Exception as e:
        print(f"Error loading datasets. Ensure the paths are correct. Error: {e}")
        return

    print(f"Loading {args.model_id} in bf16 (FSDP will shard across GPUs)...")
    # For FSDP: load model without device_map, let FSDP handle sharding
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        # Don't use device_map with FSDP - it handles placement
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Weights from notebook: [1.0, 1.5, 2.0, 0.5, 0.5, 2.0, 0.5]
    reward_fns = [
        roberta_score,
        structure_diversity_reward,
        word_pair_prompt_adherence,
        formatting,
        length_penalty,
        headline_adherence,
        coherence_penalty
    ]
    reward_weights = [1.0, 1.5, 2.0, 0.5, 0.5, 2.0, 0.5]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        report_to=args.report_to,
        num_train_epochs=args.num_train_epochs,
        use_vllm=False,
        vllm_mode=None,
        max_completion_length=args.max_completion_length,
        temperature=0.7,  # Higher temperature for more exploration
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        reward_weights=reward_weights,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,  # Keep for 32B due to memory constraints
        bf16=True,
        push_to_hub=True,
        hub_model_id=f"KonradBRG/Qwen2.5-32B-Jokester",
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
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
        model = trainer.model
        model.eval()

        prompt = "Generate the funniest possible joke that contains these two words: 'microwave', 'shoes'. Return only the joke."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            for i in range(5):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if prompt in response:
                    response = response.split(prompt)[-1].strip()
                print(f"Joke {i+1}: {response}")
                print("-" * 24)


if __name__ == "__main__":
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
    main()
