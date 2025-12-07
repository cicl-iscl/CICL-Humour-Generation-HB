import argparse
import os
import torch
import weave
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    """Main function to set up and run GRPO training for joke generation on 72B."""
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map=None, # Accelerate/DeepSpeed will handle device placement
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, 
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

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

    training_args = GRPOConfig(
        output_dir=args.output_dir, 
        report_to=args.report_to,
        num_train_epochs=args.num_train_epochs,
        use_vllm=False, 
        vllm_mode=None,
        max_completion_length=args.max_completion_length,
        temperature=0.5,
        reward_weights=reward_weights,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        gradient_checkpointing=True, # Ensure TRL knows we want this
        bf16=True,
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
    print("\n--- Example Joke Generation Post-Training ---")
    
    # Switch to eval mode
    model.eval()
    prompt = "Generate the funniest possible joke that contains these two words: 'microwave', 'shoes'. Return only the joke."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # Explicitly send to CUDA
    
    with torch.no_grad():
        for i in range(5):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            print(f"Joke {i+1}: {response}")
            print("-" * 24)

if __name__ == "__main__":
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0' 
    main()