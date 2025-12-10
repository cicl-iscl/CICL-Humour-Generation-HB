import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import XLMRobertaTokenizer
from dotenv import load_dotenv


def build_joke_dataset():
    """
    Loads English, Chinese, and Spanish joke data, processes it, and returns a single DataFrame.
    """
    eval_df = pd.read_csv("../data/labeled_jokes_full.csv")
    eval_df_zh = pd.read_csv("../data/zh_data_labeled_qwen7b.csv")
    eval_df_es = pd.read_csv("../data/es_data_labeled_llama3.1.csv")
    eval_df_zh["labels"] = eval_df_zh.score.astype(int)
    eval_df_es["labels"] = eval_df_es.score.astype(int)
    eval_df_zh = eval_df_zh[["joke", "labels"]].dropna()
    eval_df_es = eval_df_es[["joke", "labels"]].dropna()
    
    eval_df = eval_df[["joke", "labels"]].dropna()

    return pd.concat([eval_df, eval_df_zh, eval_df_es], ignore_index=True)


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


def load_datasets(model_name):
    """
    Main function to load and preprocess data, returning a DatasetDict.
    """
    eval_df = build_joke_dataset()
    train_ds, test_ds, val_ds, train_df = get_train_test_split(eval_df)

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

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
