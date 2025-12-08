import argparse
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def parse_args():
    """Parses command line arguments common to evaluation scripts."""
    parser = argparse.ArgumentParser(description="Evaluate jokes using a Hugging Face model optimized with Accelerate.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Orion-zhen/Qwen2.5-7B-Instruct-Uncensored",
        help="The Hugging Face model identifier to use for evaluation."
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
        default="zh_data_labeled.csv",
        help="Path to save the output CSV file with scores."
    )
    return parser.parse_args()

class ListDataset(Dataset):
    """
    Custom PyTorch Dataset for holding pre-formatted text prompts.
    It applies the chat template but does not tokenize here.
    """
    def __init__(self, original_list, model_name):
        self.original_list = original_list
        # The tokenizer must be loaded with the specific model's name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure the tokenizer has a pad token for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        # Format the joke into the chat template
        messages = [{"role": "user", "content": self.original_list[i]}]
        
        # apply_chat_template returns the formatted string
        # We add the assistant prefix to guide the generation
        # The exact prefix (<|im_start|>assistant or <|eot_id|>assistant) 
        # is specific to the model family (Qwen vs Llama). We'll handle 
        # the model-specific prefix extraction in the main script.
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

def get_rating(evaluation_text):
    """
    Extracts the 'rating' integer from the generated text,
    robustly handling JSON formatting issues.
    """
    # Regex to find {"rating": N} where N is 0-10
    match = re.search(r'"rating":\s*(\d{1,2})', evaluation_text)
    if match:
        rating = int(match.group(1))
        # Ensure the rating is within the valid range [0, 10]
        return min(10.0, max(0.0, float(rating)))
    
    # Fallback if no valid rating is found
    return 0.0

def collate_fn(batch, tokenizer):
    """Custom collate function for padding the batched text prompts."""
    # Tokenize the batch of text strings
    tokenized_batch = tokenizer(
        batch, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    return tokenized_batch