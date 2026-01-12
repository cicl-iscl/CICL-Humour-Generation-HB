"""
Generate joke submissions from a TSV input file using a trained model.

Usage:
    python run_submissions.py --input_tsv ../data/task-a-en.tsv --model_id KonradBRG/Qwen2.5-7B-Instruct-Jokester-English --language en --output_tsv submissions_en.tsv
    python run_submissions.py --input_tsv ../data/task-a-zh.tsv --model_id KonradBRG/Qwen2.5-7B-Instruct-Jokester-Chinese --language zh --output_tsv submissions_zh.tsv
    python run_submissions.py --input_tsv ../data/task-a-es.tsv --model_id KonradBRG/Qwen2.5-7B-Instruct-Jokester-Spanish --language es --output_tsv submissions_es.tsv
"""
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm


# Prompt constructors for each language
def construct_pair_prompt_en(w1: str, w2: str) -> str:
    return f"Generate a funny joke using these two words: '{w1}', '{w2}'."


def construct_headline_prompt_en(headline: str) -> str:
    return f"Generate a funny joke related to this headline: '{headline}'."


def construct_pair_prompt_zh(w1: str, w2: str) -> str:
    return f"用这两个词生成一个有趣的笑话：'{w1}'、'{w2}'。"


def construct_headline_prompt_zh(headline: str) -> str:
    return f"根据这个标题生成一个有趣的笑话：'{headline}'。"


def construct_pair_prompt_es(w1: str, w2: str) -> str:
    return f"Genera un chiste gracioso usando estas dos palabras: '{w1}', '{w2}'."


def construct_headline_prompt_es(headline: str) -> str:
    return f"Genera un chiste gracioso relacionado con este titular: '{headline}'."


def get_prompt_functions(language: str):
    """Get the prompt constructor functions for a language."""
    if language == "en":
        return construct_pair_prompt_en, construct_headline_prompt_en
    elif language == "zh":
        return construct_pair_prompt_zh, construct_headline_prompt_zh
    elif language == "es":
        return construct_pair_prompt_es, construct_headline_prompt_es
    else:
        raise ValueError(f"Unknown language: {language}")


def load_and_prepare_data(input_tsv: str, language: str) -> pd.DataFrame:
    """Load TSV file and create prompts for each row."""
    df = pd.read_csv(input_tsv, delimiter="\t")

    pair_prompt_fn, headline_prompt_fn = get_prompt_functions(language)

    prompts = []
    for _, row in df.iterrows():
        w1 = row.get("word1", "-")
        w2 = row.get("word2", "-")
        headline = row.get("headline", "-")

        # Handle NaN values
        if pd.isna(w1):
            w1 = "-"
        if pd.isna(w2):
            w2 = "-"
        if pd.isna(headline):
            headline = "-"

        if w1 != "-" and w2 != "-":
            prompts.append(pair_prompt_fn(w1, w2))
        elif headline != "-":
            prompts.append(headline_prompt_fn(headline))
        else:
            # Fallback - should not happen with valid data
            prompts.append("Generate a funny joke.")

    df["prompt"] = prompts
    return df


def generate_jokes(
    df: pd.DataFrame,
    model_id: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    batch_size: int = 1,
    num_return_sequences: int = 1,
) -> list:
    """Generate jokes using the specified model."""

    print(f"Loading model: {model_id}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if device == "cuda" else None,
    )

    print(f"Generating jokes for {len(df)} prompts...")

    generations = []
    prompts = df["prompt"].tolist()

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        try:
            outputs = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                return_full_text=False,  # Only return the generated part
            )

            # Extract the generated text
            generated_text = outputs[0]["generated_text"].strip()

            # Clean up the generation (remove common artifacts)
            generated_text = clean_generation(generated_text)

            generations.append(generated_text)

        except Exception as e:
            print(f"Error generating for prompt {i}: {e}")
            generations.append("")

    return generations


def clean_generation(text: str) -> str:
    """Clean up generated text by removing common artifacts."""
    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove common prefixes that models sometimes add
    prefixes_to_remove = [
        "Here's a joke:",
        "Here is a joke:",
        "Sure, here's a joke:",
        "Sure! Here's a joke:",
        "Joke:",
        "Answer:",
    ]

    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove quotes if the entire text is quoted
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Take only the first joke if multiple are generated (split by common separators)
    separators = ["\n\n", "\n---", "---", "Joke 2:", "2.", "Here's another"]
    for sep in separators:
        if sep in text:
            text = text.split(sep)[0].strip()

    return text


def save_submissions(df: pd.DataFrame, generations: list, output_tsv: str):
    """Save the submissions to a TSV file."""
    submissions = pd.DataFrame({
        "id": df["id"],
        "text": generations,
    })

    # Save as TSV
    submissions.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved {len(submissions)} submissions to {output_tsv}")

    # Print some examples
    print("\nExample submissions:")
    for i in range(min(3, len(submissions))):
        print(f"  {submissions.iloc[i]['id']}: {submissions.iloc[i]['text'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate joke submissions from TSV input")
    parser.add_argument(
        "--input_tsv",
        type=str,
        required=True,
        help="Path to input TSV file with columns: id, word1, word2, headline",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to the trained model",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en", "zh", "es"],
        help="Language for prompt construction",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        required=True,
        help="Path to output TSV file with columns: id, text",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)",
    )

    args = parser.parse_args()

    # Load and prepare data
    print(f"Loading data from {args.input_tsv}...")
    df = load_and_prepare_data(args.input_tsv, args.language)
    print(f"Loaded {len(df)} rows")

    # Generate jokes
    generations = generate_jokes(
        df,
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    # Save submissions
    save_submissions(df, generations, args.output_tsv)

    print("\nDone!")


if __name__ == "__main__":
    main()
