import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from modeling_custom import HierarchicalClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    XLMRobertaTokenizer, XLMRobertaPreTrainedModel, XLMRobertaModel,
    XLMRobertaConfig, AutoConfig, Trainer, TrainingArguments
)
import os



def build_joke_dataset():
    """
    Loads English and Chinese joke data, processes it, and returns a single DataFrame.
    """

    eval_df = pd.read_csv("../data/labeled_jokes_full.csv")
    eval_df_zh = pd.read_csv("../data/zh_data_labeled.csv")
    eval_df_zh["labels"] = eval_df_zh.score.astype(int)
    eval_df_zh = eval_df_zh[["joke", "labels"]].dropna()
    
    return pd.concat([eval_df, eval_df_zh], ignore_index=True)


def get_train_test_split(eval_df, test_size=0.2, random_state=42):
    """
    Splits the main DataFrame into train, test, and validation sets and converts them
    into Hugging Face Dataset objects.
    """
    train_df, test_val_df = train_test_split(
        eval_df, test_size=test_size, random_state=random_state, stratify=eval_df.labels
    )
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

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)


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
    rmse = mean_squared_error(labels, preds, squared=False) 
    mae = mean_absolute_error(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"rmse": rmse, "accuracy": acc, "mae": mae}


def get_cross_entropy_weights(train_ds):
    """
    Calculates class weights for binary and child classification tasks to handle
    class imbalance using the formula: weight_i = total_count / (num_classes * count_i).
    """
    labels = np.array(train_ds["labels"])
    
    binary_counts = np.array([(labels == 0).sum(), (labels != 0).sum()], dtype=float)
    binary_weights = (binary_counts.sum() / (2 * binary_counts))
    binary_weights = torch.tensor(binary_weights, dtype=torch.float)

    child_labels = labels[labels != 0]    
    num_child_classes = np.max(child_labels) 
    child_counts = np.array([(child_labels == c).sum() for c in range(1, num_child_classes + 1)], dtype=float)

    child_weights = (child_counts.sum() / (num_child_classes * child_counts))
    child_weights = torch.tensor(child_weights, dtype=torch.float)

    return binary_weights, child_weights



def setup_custom_config_and_save(model, tokenizer, train_df, binary_weights, child_weights, output_dir):
    """
    Registers and updates the model's configuration for saving/sharing with
    custom attributes and auto_map for the custom model class.
    """
    
    from modeling_custom import HierarchicalConfig

    HierarchicalClassifier.config_class = HierarchicalConfig
    AutoConfig.register("xlm-roberta-joke-rater", HierarchicalConfig)
    AutoModelForSequenceClassification.register(HierarchicalConfig, HierarchicalClassifier)

    current_config_dict = model.config.to_dict()
    model.config = HierarchicalConfig(**current_config_dict)

    n_labels = len(train_df.labels.unique())
    model.config.num_child_labels = n_labels - 1
    model.config.class_weights_binary = binary_weights.tolist()
    model.config.class_weights_child = child_weights.tolist()

    unique_labels = sorted(train_df["labels"].unique())
    id2label = {int(i): int(label) for i, label in enumerate(unique_labels)} # Use int for labels
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    model.config.auto_map = {
        "AutoConfig": "modeling_custom.HierarchicalConfig",
        "AutoModelForSequenceClassification": "modeling_custom.HierarchicalClassifier"
    }

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Custom config and model registered and saved successfully.")
    print(f"Generated id2label: {id2label}")



if __name__ == "__main__":
    load_dotenv()
    MODEL_NAME = "FacebookAI/xlm-roberta-large"
    OUTPUT_DIR = "./xlm-roberta-joke-rater"
    HUB_MODEL_ID = "KonradBRG/xlm-roberta-joke-rater"
    
    # Data Loading and Preprocessing
    datasets, tokenizer, train_df = load_datasets(MODEL_NAME)
    train_ds = datasets["train"]
    val_ds = datasets["validation"]
    test_ds = datasets["test"]
    
    # Calculate Class Weights
    binary_weights, child_weights = get_cross_entropy_weights(train_ds)
    n_labels = len(train_df.labels.unique())
   
    # Initialize Custom Model
    model = HierarchicalClassifier.from_pretrained(
        MODEL_NAME,
        num_child_labels=n_labels - 1, # 10 classes (1 to 10)
        class_weights_binary=binary_weights,
        class_weights_child=child_weights
    )

    # Setup Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        num_train_epochs=56,
        learning_rate=2e-5,
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

    setup_custom_config_and_save(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        binary_weights=binary_weights,
        child_weights=child_weights,
        output_dir=OUTPUT_DIR
    )
    
    print(f"Final model and tokenizer saved to {OUTPUT_DIR}")