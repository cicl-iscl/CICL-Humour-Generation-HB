import os
import pandas as pd
from tqdm import tqdm

import kagglehub
from datasets import load_dataset
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

# Import shared utilities
from eval_utils import parse_args, get_rating

# --- 1. CONFIGURATION ---

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


class JokeDataset(Dataset):
    """Simple dataset for joke evaluation."""

    def __init__(self, prompts, tokenizer):
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        messages = [{"role": "user", "content": self.prompts[i]}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def collate_fn(batch, tokenizer):
    return tokenizer(batch, padding=True, truncation=True, return_tensors="pt")


# --- 2. DATA LOADING ---


def load_and_combine_jokes():
    """Loads jokes from Kaggle and Hugging Face datasets."""
    print("Loading and combining datasets...")

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

    # 1. Load Data
    jokes_combined = load_and_combine_jokes()
    eval_prompts = [PROMPT_ES.format(joke) for joke in jokes_combined]

    # 2. Setup Model and Tokenizer (simple single/multi-GPU with device_map)
    print(f"Initializing model: {args.model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = (
        "left"  # Required for batched generation with decoder-only models
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # 3. Prepare DataLoader
    eval_dataset = JokeDataset(eval_prompts, tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    # 4. Inference Loop
    print(f"Starting inference with batch size {args.batch_size}...")

    evals = []
    generation_kwargs = {
        "max_new_tokens": 150,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for batch in tqdm(eval_dataloader, desc="Evaluating Jokes"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs
            )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        generated_texts = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False
        )

        cleaned_texts = [
            (
                text.split(ASSISTANT_PREFIX)[-1].strip()
                if ASSISTANT_PREFIX in text
                else text.strip()
            )
            for text in generated_texts
        ]

        evals.extend(cleaned_texts)

    # 5. Process and Save Results
    print("Processing and saving results...")

    eval_df = pd.DataFrame()
    eval_df["joke"] = jokes_combined
    eval_df["model_raw_output"] = evals
    eval_df["score"] = [get_rating(e) for e in evals]

    eval_df.to_csv(args.output_file, index=False)
    print(f"Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
