import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def load_english_data(data_dir: str = "../data"):
    """Load English joke data from combined_jokes_full.csv."""
    df = pd.read_csv(f"{data_dir}/combined_jokes_full.csv")
    df = df[["joke", "labels"]].dropna()
    df["labels"] = df["labels"].astype(int)
    return df


def load_chinese_data(data_dir: str = "../data"):
    """Load Chinese joke data from zh_jokes_combined.tsv."""
    df = pd.read_csv(f"{data_dir}/zh_jokes_combined.tsv", sep='\t')
    df["labels"] = df.average_weighted.astype(int)
    df = df[["joke", "labels"]].dropna()
    return df


def load_spanish_data(data_dir: str = "../data"):
    """Load Spanish joke data from es_data_labeled_llama3.1.csv."""
    df = pd.read_csv(f"{data_dir}/es_data_labeled_llama3.1.csv")
    df["labels"] = df["score"].astype(int)
    df = df[["joke", "labels"]].dropna()
    return df


def build_joke_dataset(language: str = "all", data_dir: str = "../data"):
    """
    Loads joke data for specified language(s) and returns a single DataFrame.

    Args:
        language: One of "en", "zh", "es", or "all" for multilingual
        data_dir: Path to data directory

    Returns:
        DataFrame with columns ['joke', 'labels']
    """
    if language == "en":
        return load_english_data(data_dir)
    elif language == "zh":
        return load_chinese_data(data_dir)
    elif language == "es":
        return load_spanish_data(data_dir)
    elif language == "all":
        # Load all three languages for multilingual training
        eval_df_en = load_english_data(data_dir)
        eval_df_zh = load_chinese_data(data_dir)
        eval_df_es = load_spanish_data(data_dir)
        return pd.concat([eval_df_en, eval_df_zh, eval_df_es], ignore_index=True)
    else:
        raise ValueError(f"Unknown language: {language}. Use 'en', 'zh', 'es', or 'all'.")


def get_train_test_split(eval_df, test_size=0.2, random_state=42):
    """
    Splits the main DataFrame into train, test, and validation sets and converts them
    into Hugging Face Dataset objects.
    """
    # Stratification ensures equal distribution of joke scores (labels 0-10) across splits
    train_df, test_val_df = train_test_split(
        eval_df, test_size=test_size, random_state=random_state, stratify=eval_df.labels
    )
    # Split the remaining 20% into 10% test and 10% validation
    test_df, val_df = train_test_split(
        test_val_df, test_size=0.5, random_state=random_state
    )
    print(
        f"Dataset sizes: Train={len(train_df)}, Test={len(test_df)}, Validation={len(val_df)}"
    )

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, test_ds, val_ds, train_df


def load_datasets(model_name: str, language: str = "all", data_dir: str = "../data"):
    """
    Main function to load and preprocess data, returning a DatasetDict.

    Args:
        model_name: HuggingFace model name for tokenizer
        language: One of "en", "zh", "es", or "all"
        data_dir: Path to data directory
    """
    eval_df = build_joke_dataset(language=language, data_dir=data_dir)
    print(f"Loaded {len(eval_df)} jokes for language: {language}")

    train_ds, test_ds, val_ds, train_df = get_train_test_split(eval_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["joke"], truncation=True, padding="max_length", max_length=128
        )

    # Tokenize and map the datasets
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Remove temporary columns used by pandas/map (if they exist)
    for col in ["__index_level_0__"]:
        if col in train_ds.column_names:
            train_ds = train_ds.remove_columns([col])
        if col in test_ds.column_names:
            test_ds = test_ds.remove_columns([col])
        if col in val_ds.column_names:
            val_ds = val_ds.remove_columns([col])

    datasets = DatasetDict({"train": train_ds, "test": test_ds, "validation": val_ds})

    return datasets, tokenizer, train_df
