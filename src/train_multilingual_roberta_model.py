import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from modeling_custom import HierarchicalClassifier
from modeling_custom import HierarchicalConfig
from sklearn.metrics import root_mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    XLMRobertaTokenizer, 
    XLMRobertaConfig, AutoConfig, Trainer, TrainingArguments
)


def build_joke_dataset():
    """
    Loads English and Chinese joke data, processes it, and returns a single DataFrame.
    """
    eval_df = pd.read_csv("../data/labeled_jokes_full.csv")
    eval_df_zh = pd.read_csv("../data/zh_data_labeled.csv")
    eval_df_zh["labels"] = eval_df_zh.score.astype(int)
    eval_df_zh = eval_df_zh[["joke", "labels"]].dropna() 
    eval_df = eval_df[["joke", "labels"]].dropna()
    
    return pd.concat([eval_df, eval_df_zh], ignore_index=True)


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
    print(f"Dataset sizes: Train={len(train_df)}, Test={len(test_df)}, Validation={len(val_df)}")
    
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
            batch["joke"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )

    # Tokenize and map the datasets
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Remove temporary columns used by pandas/map
    train_ds = train_ds.remove_columns(['__index_level_0__'])
    test_ds = test_ds.remove_columns(['__index_level_0__'])
    val_ds = val_ds.remove_columns(['__index_level_0__'])


    datasets = DatasetDict({
        "train": train_ds,
        "test": test_ds,
        "validation": val_ds
    })
    
    return datasets, tokenizer, train_df


def compute_metrics(eval_pred):
    """
    Computes RMSE, MAE, and Accuracy for the model's predictions.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    rmse = root_mean_squared_error(labels, preds, squared=False) 
    mae = mean_absolute_error(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"rmse": rmse, "accuracy": acc, "mae": mae}


def get_cross_entropy_weights(train_ds):
    """
    Calculates class weights for binary and child classification tasks to handle
    class imbalance using the formula: weight_i = total_count / (num_classes * count_i).
    """
    labels = np.array(train_ds["labels"])
    
    # 1. Binary Weights (0 vs. 1-10)
    binary_counts = np.array([(labels == 0).sum(), (labels != 0).sum()], dtype=float)
    binary_weights = (binary_counts.sum() / (2 * binary_counts))
    binary_weights = torch.tensor(binary_weights, dtype=torch.float)

    # 2. Child Weights (1 to 10, only for non-zero labels)
    child_labels = labels[labels != 0]    
    num_child_classes = 10 # Labels 1 through 10
    child_counts = np.array([(child_labels == c).sum() for c in range(1, num_child_classes + 1)], dtype=float)

    child_weights = (child_counts.sum() / (num_child_classes * child_counts))
    child_weights = torch.tensor(child_weights, dtype=torch.float)

    return binary_weights, child_weights


def setup_custom_config_and_save(model, tokenizer, train_df, binary_weights, child_weights, output_dir):
    """
    Registers and updates the model's configuration for saving/sharing with
    custom attributes and auto_map for the custom model class.
    """

    HierarchicalClassifier.config_class = HierarchicalConfig
    AutoConfig.register("xlm-roberta-joke-rater", HierarchicalConfig)
    AutoModelForSequenceClassification.register(HierarchicalConfig, HierarchicalClassifier)

    current_config_dict = model.config.to_dict()
    # Instantiate the custom config using the base model's config
    model.config = HierarchicalConfig(**current_config_dict)

    n_labels = len(train_df.labels.unique()) # Should be 11 (0-10)
    model.config.num_child_labels = n_labels - 1 # 10
    model.config.class_weights_binary = binary_weights.tolist()
    model.config.class_weights_child = child_weights.tolist()

    unique_labels = sorted(train_df["labels"].unique()) # [0, 1, ..., 10]
    id2label = {int(i): int(label) for i, label in enumerate(unique_labels)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = n_labels # Should be 11

    # Add auto_map for custom class serialization
    model.config.auto_map = {
        "AutoConfig": "modeling_custom.HierarchicalConfig",
        "AutoModelForSequenceClassification": "modeling_custom.HierarchicalClassifier"
    }

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nCustom config and model registered and saved successfully.")
    print(f"Generated id2label: {id2label}")


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a Hierarchical XLM-RoBERTa Classifier for Joke Rating.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="FacebookAI/xlm-roberta-large",
        help="Pre-trained model to use as the base (e.g., 'FacebookAI/xlm-roberta-large')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./xlm-roberta-joke-rater",
        help="Directory to save the trained model and tokenizer."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="KonradBRG/xlm-roberta-joke-rater",
        help="Hugging Face model ID for pushing to the Hub."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=56,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the Trainer."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU for training."
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    load_dotenv()
    
    # Use parsed arguments
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    HUB_MODEL_ID = args.hub_model_id
    NUM_TRAIN_EPOCHS = args.num_train_epochs
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.per_device_train_batch_size
    
    print(f"--- Starting Training ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Data Loading and Preprocessing
    datasets, tokenizer, train_df = load_datasets(MODEL_NAME)
    train_ds = datasets["train"]
    val_ds = datasets["validation"]
    test_ds = datasets["test"]
    
    # Calculate Class Weights
    binary_weights, child_weights = get_cross_entropy_weights(train_ds)
    n_labels = len(train_df.labels.unique())
   
    # Initialize Custom Model
    try:
        from modeling_custom import HierarchicalClassifier
        model = HierarchicalClassifier.from_pretrained(
            MODEL_NAME,
            num_child_labels=n_labels - 1, # 10 classes (1 to 10)
            class_weights_binary=binary_weights,
            class_weights_child=child_weights
        )
    except ImportError:
        print("FATAL ERROR: HierarchicalClassifier could not be imported. Ensure modeling_custom.py is correct.")
        exit(1)


    # Setup Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=8,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
    )

    # Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on Test Set
    test_results = trainer.predict(test_ds)
    print("--- Test results ---")
    print(test_results.metrics)

    # Save custom config and model
    setup_custom_config_and_save(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        binary_weights=binary_weights,
        child_weights=child_weights,
        output_dir=OUTPUT_DIR
    )
    
    print(f"\nFinal model and tokenizer saved to {OUTPUT_DIR}")