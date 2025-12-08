import argparse
import json
import re
import os
import pandas as pd
from tqdm import tqdm

import kagglehub
from huggingface_hub import login, HfFolder
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

# --- 1. ARGPARSE SETUP ---
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Spanish jokes using a Hugging Face model optimized with Accelerate.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The Hugging Face model identifier to use for evaluation (e.g., 'meta-llama/Llama-3.1-8B-Instruct')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for accelerated inference."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="es_data_labeled.csv",
        help="Path to save the output CSV file with scores."
    )
    return parser.parse_args()

# --- 2. DATASET PREPARATION ---

class ListDataset(Dataset):
    """
    Custom PyTorch Dataset for holding pre-formatted text prompts.
    It applies the chat template but does not tokenize here,
    as tokenization will be handled by the DataLoader/Pipeline.
    """
    def __init__(self, original_list, tokenizer):
        self.original_list = original_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        # Format the joke into the chat template
        messages = [{"role": "user", "content": self.original_list[i]}]
        
        # apply_chat_template returns the formatted string
        # We add the assistant prefix to guide the generation (needed for instruction-tuned models)
        return self.tokenizer.apply_chat_template(messages, tokenize=False) + "<|eot_id|>assistant\n"

def load_and_combine_jokes():
    """Loads jokes from Kaggle and Hugging Face datasets."""
    print("📥 Loading and combining datasets...")
    
    # Download latest version of the Kaggle dataset
    try:
        path = kagglehub.dataset_download("bachrr/haha-2019")
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Please ensure you have configured your Kaggle API key.")
        raise

    df_train = pd.read_csv(os.path.join(path, "haha_2019_train.csv"))
    
    # Load Spanish jokes from Hugging Face
    ds = load_dataset("mrm8488/CHISTES_spanish_jokes")
    chistes_df = ds["train"].to_pandas()

    jokes_combined = df_train.text.to_list() + chistes_df.text.to_list()
    print(f"Total jokes loaded: {len(jokes_combined)}")
    return jokes_combined

# --- 3. EVALUATION LOGIC ---

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

def get_rating(evaluation_text):
    """
    Extracts the 'rating' integer from the generated text,
    robustly handling JSON formatting issues.
    """
    # Regex to find {"rating": N} where N is 0-10, even if the JSON is malformed/incomplete
    match = re.search(r'"rating":\s*(\d{1,2})', evaluation_text)
    if match:
        rating = int(match.group(1))
        # Ensure the rating is within the valid range [0, 10]
        return min(10.0, max(0.0, float(rating)))
    
    # Fallback if no valid rating is found
    return 0.0

def main():
    """Main function to run the joke evaluation."""
    args = parse_args()
    
    # Ensure Hugging Face login is performed
    if not HfFolder.get_token():
        print("🔑 Hugging Face login required. Please log in interactively or set the HUGGING_FACE_HUB_TOKEN environment variable.")
        login()

    # 1. Load Data
    jokes_combined = load_and_combine_jokes()
    
    # Pre-format the jokes with the detailed prompt
    eval_prompts = [PROMPT_ES.format(joke) for joke in jokes_combined]

    # 2. Setup Model, Tokenizer, and Accelerator
    print(f"🤖 Initializing model: {args.model_name}...")
    
    # Initialize Accelerator
    # Accelerate handles device placement and distributed setup (if applicable)
    accelerator = Accelerator()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Set padding token for batching, as it's not always set by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if accelerator.is_available() else torch.float32,
        device_map={"": accelerator.device},
    )

    # 3. Prepare Dataset and DataLoader
    eval_dataset = ListDataset(eval_prompts, tokenizer)
    
    # We use a standard DataLoader for batching, which Accelerator will optimize
    def collate_fn(batch):
        """Custom collate function for padding the batched text prompts."""
        # Tokenize the batch of text strings
        tokenized_batch = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return tokenized_batch

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # Prepare for acceleration
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # 4. Inference Loop
    print(f"🚀 Starting accelerated inference with batch size {args.batch_size}...")
    
    evals = []
    model.eval()
    
    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": 150, # Sufficient length for the JSON output and reason
        "do_sample": False,
        "temperature": 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    # The model now only generates the ASSISTANT's response given the input IDs
    for batch in tqdm(eval_dataloader, desc="Evaluating Jokes"):
        # Move inputs to the correct device
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        
        # Perform generation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        
        # Decode the generated text
        # Skip the input tokens to get only the new generation
        generated_tokens = outputs[:, input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        
        # Append results
        evals.extend(generated_texts)

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