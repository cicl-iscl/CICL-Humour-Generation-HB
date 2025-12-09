import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Import shared utilities
from eval_utils import parse_args, ListDataset, get_rating, collate_fn

# --- 1. PROMPT & MODEL CONFIG ---

PROMPT_ZH = """
观察类笑话是通过喜剧视角审视日常事物或情境。它们涵盖几乎人人熟悉的主题，甚至涉及生活的最琐碎细节。
而轶事类幽默则源自喜剧演员的个人经历，因观众能产生共鸣而广受欢迎。
你既欣赏观察类与轶事类幽默，也钟爱冷笑话和反讽。
你懂得欣赏妙趣横生的笑话，但要让你发笑也绝非易事。
你的任务是按0至10分评分：0分代表毫无趣味，10分代表爆笑至极。平庸笑话通常得5分。
9分或10分极为罕见，仅授予顶尖笑话。因此8分已是相当高的评价，请勿轻易给予。
请仅返回包含有效JSON的字段：`rating`（整数形式的评分）和`reason`（评分理由）。
笑话内容如下：
{}
"""

# The exact prefix for Qwen models
ASSISTANT_PREFIX = "<|im_start|>assistant\n" 

# --- 2. MAIN EXECUTION ---

def main():
    """Main function to run the joke evaluation."""
    args = parse_args()
    
    # 1. Load Data
    print("📥 Loading jokes from zh_jokes.csv...")
    try:
        # NOTE: Ensure zh_jokes.csv is in the current working directory or provide the full path
        zh_joke_df = pd.read_csv("zh_jokes.csv")
    except FileNotFoundError:
        print("Error: zh_jokes.csv not found. Please ensure it's in the correct directory.")
        return

    jokes_combined = zh_joke_df.joke.to_list()
    print(f"Total jokes loaded: {len(jokes_combined)}")
    
    # Pre-format the jokes with the detailed prompt
    eval_prompts = [PROMPT_ZH.format(joke) for joke in jokes_combined]

    # 2. Setup Model, Tokenizer, and Accelerator
    print(f"🤖 Initializing model: {args.model_name}...")

    accelerator = Accelerator(mixed_precision="bf16")

    # Initialize dataset and get the tokenizer used within it
    eval_dataset = ListDataset(eval_prompts, args.model_name)
    tokenizer = eval_dataset.tokenizer

    # 3. Prepare DataLoader
    data_loader_collate_fn = lambda batch: collate_fn(batch, tokenizer)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_loader_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model - let Accelerate handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Prepare for acceleration - this handles device placement
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # 4. Inference Loop
    print(f"🚀 Starting accelerated inference with batch size {args.batch_size}...")
    
    evals = []
    model.eval()
    
    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": 150, 
        "do_sample": False,
        "temperature": 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    for batch in tqdm(eval_dataloader, desc="Evaluating Jokes"):
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        
        generated_tokens = outputs[:, input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        
        # Qwen models include the full prompt and a special start token in the output.
        # We need to clean this to get the raw JSON response.
        cleaned_texts = [
            text.split(ASSISTANT_PREFIX)[-1].strip() if ASSISTANT_PREFIX in text else text.strip()
            for text in generated_texts
        ]
        
        evals.extend(cleaned_texts)

    # 5. Process and Save Results
    print("📝 Processing and saving results...")
    
    eval_df = pd.DataFrame()
    eval_df["joke"] = jokes_combined
    eval_df["model_raw_output"] = evals
    eval_df["score"] = [get_rating(e) for e in evals]

    # Save the final dataframe
    eval_df.to_csv(args.output_file, index=False)
    print(f"✅ Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()