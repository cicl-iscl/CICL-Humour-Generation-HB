"""
Ablation study comparing HierarchicalClassifier with different backbones:
- xlm-roberta-large (fine-tuned vs not fine-tuned)
- mdeberta-v3-base (fine-tuned vs not fine-tuned)

Outputs a table with Accuracy and MAE on the test split.
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
from datasets import DatasetDict
from tabulate import tabulate

from joke_rater.preprocessing import build_joke_dataset, get_train_test_split
from joke_rater.modeling_custom import (
    HierarchicalClassifier,
    HierarchicalDebertaClassifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for joke rater models")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs for fine-tuned models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Directory to save results",
    )
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Compute accuracy and MAE."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {"accuracy": acc, "mae": mae}


def get_cross_entropy_weights(train_ds):
    """Calculate class weights for imbalanced data."""
    labels = np.array(train_ds["labels"])

    # Binary weights (0 vs 1-10)
    binary_counts = np.array([(labels == 0).sum(), (labels != 0).sum()], dtype=float)
    binary_weights = binary_counts.sum() / (2 * binary_counts)
    binary_weights = torch.tensor(binary_weights, dtype=torch.float)

    # Child weights (1 to 10)
    child_labels = labels[labels != 0]
    num_child_classes = 10
    child_counts = np.array(
        [(child_labels == c).sum() for c in range(1, num_child_classes + 1)],
        dtype=float,
    )
    # Avoid division by zero
    child_counts = np.maximum(child_counts, 1)
    child_weights = child_counts.sum() / (num_child_classes * child_counts)
    child_weights = torch.tensor(child_weights, dtype=torch.float)

    return binary_weights, child_weights


def prepare_datasets(model_name: str):
    """Load and tokenize datasets for a given model."""
    eval_df = build_joke_dataset()
    train_ds, test_ds, val_ds, train_df = get_train_test_split(eval_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["joke"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Remove pandas index column if present
    for col in ["__index_level_0__"]:
        if col in train_ds.column_names:
            train_ds = train_ds.remove_columns([col])
        if col in test_ds.column_names:
            test_ds = test_ds.remove_columns([col])
        if col in val_ds.column_names:
            val_ds = val_ds.remove_columns([col])

    datasets = DatasetDict({
        "train": train_ds,
        "test": test_ds,
        "validation": val_ds,
    })

    return datasets, tokenizer, train_df


def evaluate_model(model, test_ds, tokenizer, batch_size: int = 32):
    """Evaluate a model on the test set without training."""
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=batch_size,
        report_to="none",
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    results = trainer.predict(test_ds)
    return results.metrics


def train_and_evaluate(
    model,
    datasets,
    tokenizer,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str,
):
    """Train a model and evaluate on test set."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        bf16=torch.cuda.is_available(),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    results = trainer.predict(datasets["test"])
    return results.metrics


def run_ablation(args):
    """Run the full ablation study."""
    results = []

    # Model configurations
    configs = [
        {
            "name": "xlm-roberta-large",
            "model_class": HierarchicalClassifier,
            "model_id": "xlm-roberta-large",
        },
        {
            "name": "mdeberta-v3-base",
            "model_class": HierarchicalDebertaClassifier,
            "model_id": "microsoft/mdeberta-v3-base",
        },
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {config['name']}")
        print("=" * 60)

        # Prepare datasets
        datasets, tokenizer, train_df = prepare_datasets(config["model_id"])
        binary_weights, child_weights = get_cross_entropy_weights(datasets["train"])
        n_labels = len(train_df.labels.unique())

        # --- No Fine-tuning ---
        print(f"\n[{config['name']}] Evaluating WITHOUT fine-tuning...")
        model_no_ft = config["model_class"].from_pretrained(
            config["model_id"],
            num_child_labels=n_labels - 1,
            class_weights_binary=binary_weights,
            class_weights_child=child_weights,
        )

        metrics_no_ft = evaluate_model(
            model_no_ft,
            datasets["test"],
            tokenizer,
            args.batch_size,
        )

        results.append({
            "Model": config["name"],
            "Fine-tuned": "No",
            "Accuracy": f"{metrics_no_ft['test_accuracy']:.4f}",
            "MAE": f"{metrics_no_ft['test_mae']:.4f}",
        })

        # Free memory
        del model_no_ft
        torch.cuda.empty_cache()

        # --- With Fine-tuning ---
        print(f"\n[{config['name']}] Training WITH fine-tuning...")
        model_ft = config["model_class"].from_pretrained(
            config["model_id"],
            num_child_labels=n_labels - 1,
            class_weights_binary=binary_weights,
            class_weights_child=child_weights,
        )

        metrics_ft = train_and_evaluate(
            model_ft,
            datasets,
            tokenizer,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=f"{args.output_dir}/{config['name'].replace('/', '_')}_finetuned",
        )

        results.append({
            "Model": config["name"],
            "Fine-tuned": "Yes",
            "Accuracy": f"{metrics_ft['test_accuracy']:.4f}",
            "MAE": f"{metrics_ft['test_mae']:.4f}",
        })

        # Free memory
        del model_ft
        torch.cuda.empty_cache()

    # Print results table
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(tabulate(results, headers="keys", tablefmt="github"))

    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.output_dir}/ablation_results.csv", index=False)
    print(f"\nResults saved to {args.output_dir}/ablation_results.csv")

    return results


if __name__ == "__main__":
    args = parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    run_ablation(args)
