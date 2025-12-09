import argparse
import os
import pandas as pd
from tqdm import tqdm

import kagglehub
from huggingface_hub import login, HfFolder
from datasets import load_dataset
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Import shared utilities
from eval_utils import parse_args, ListDataset, get_rating, collate_fn

# --- 1. CONFIGURATION ---

# The specific assistant prefix for Llama/Meta models
ASSISTANT_PREFIX = "<|eot_id|>assistant\n"

PROMPT_ES = """
Los chistes observacionales son un análisis de cosas o situaciones cotidianas desde una perspectiva cómica. Abarcan temas familiares para casi todo el mundo, incluso los aspectos más triviales de la vida.
El humor anecdótico, sin embargo, se basa en la vida personal del cómico y es popular entre el público porque este se identifica con sus historias. 
Eres una persona que disfruta del humor observacional y anecdótico, así como de los chistes de una sola línea y la ironía. 
Aprecias los chistes divertidos, pero tampoco es fácil hacerte reír.
Tu tarea consiste en puntuar un chiste en una escala del 0 al 10, donde 0 significa que no es nada divertido y 10 significa que es realmente gracioso. Un chiste mediocre suele obtener un 5. 
Una puntuación de 9 o 10 es muy poco frecuente y se reserva solo para los mejores chistes. Por lo tanto, un 8 se considera una puntuación muy buena y no debes ser demasiado generoso con ella.
Solo debes devolver un JSON válido con los campos «rating» (que contiene tu puntuación, como un número entero) y «reason», que justifica tu respuesta.
El chiste es:
{}
"""

# --- 2. DATA LOADING ---

def load_and_combine_jokes():
    """Loads jokes from Kaggle and Hugging Face datasets."""
    print("📥 Loading and combining datasets...")
    
    try:
        path = kagglehub.dataset_download("bachrr/haha-2019")
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Please ensure you have configured your Kaggle API key.")
        raise

    df_train = pd.read_csv(os.path.join(path, "haha_2019_train.csv"))
    
    ds = load_dataset("mrm8488/CHISTES_spanish_jokes")
    chistes_df = ds["train"].to_pandas()

    jokes_combined = df_train.text.to_list() + chistes_df.text.to_list()
    print(f"Total jokes loaded: {len(jokes_combined)}")
    return jokes_combined

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to run the joke evaluation."""
    args = parse_args()
    
    if not HfFolder.get_token():
        print("🔑 Hugging Face login required. Please log in interactively or set the HUGGING_FACE_HUB_TOKEN environment variable.")
        login()

    # 1. Load Data
    jokes_combined = load_and_combine_jokes()
    
    # Pre-format the jokes with the detailed prompt
    eval_prompts = [PROMPT_ES.format(joke) for joke in jokes_combined]

    # 2. Setup Model, Tokenizer, and Accelerator
    print(f"🤖 Initializing model: {args.model_name}...")

    accelerator = Accelerator(mixed_precision="bf16")

    # We initialize the dataset here which loads the tokenizer and sets the prefix
    eval_dataset = ListDataset(eval_prompts, args.model_name, ASSISTANT_PREFIX)
    tokenizer = eval_dataset.tokenizer

    # Initialize model - let Accelerate handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

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
        
        # Clean the generated text (remove the assistant prefix if it was repeated)
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

    eval_df.to_csv(args.output_file, index=False)
    print(f"✅ Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()