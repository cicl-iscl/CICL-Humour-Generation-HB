import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Import shared utilities
from eval_utils import parse_args, get_rating

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

ASSISTANT_PREFIX = "<|im_start|>assistant\n"


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


# --- 2. MAIN EXECUTION ---


def main():
    """Main function to run the joke evaluation."""
    args = parse_args()

    # 1. Load Data
    print("Loading jokes from zh_jokes.csv...")
    try:
        zh_joke_df = pd.read_csv("../data/zh_jokes.csv")
    except FileNotFoundError:
        print("Error: zh_jokes.csv not found.")
        return

    jokes_combined = zh_joke_df.joke.to_list()
    print(f"Total jokes loaded: {len(jokes_combined)}")

    eval_prompts = [PROMPT_ZH.format(joke) for joke in jokes_combined]

    # 2. Setup Model and Tokenizer (simple single-GPU setup)
    print(f"Initializing model: {args.model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = (
        "left"  # Required for batched generation with decoder-only models
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
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
