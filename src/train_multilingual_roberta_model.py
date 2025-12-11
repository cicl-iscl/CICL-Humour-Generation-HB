import os
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error, accuracy_score, mean_absolute_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from joke_rater.cli import parse_args
from joke_rater.preprocessing import load_datasets
from joke_rater.modeling_custom import HierarchicalConfig, HierarchicalClassifier


def compute_metrics(eval_pred):
    """
    Computes RMSE, MAE, and Accuracy for the model's predictions.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    rmse = root_mean_squared_error(labels, preds)
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
    binary_weights = binary_counts.sum() / (2 * binary_counts)
    binary_weights = torch.tensor(binary_weights, dtype=torch.float)

    # 2. Child Weights (1 to 10, only for non-zero labels)
    child_labels = labels[labels != 0]
    num_child_classes = 10  # Labels 1 through 10
    child_counts = np.array(
        [(child_labels == c).sum() for c in range(1, num_child_classes + 1)],
        dtype=float,
    )

    child_weights = child_counts.sum() / (num_child_classes * child_counts)
    child_weights = torch.tensor(child_weights, dtype=torch.float)

    return binary_weights, child_weights


def setup_custom_config_and_save(
    model, tokenizer, train_df, binary_weights, child_weights, output_dir
):
    """
    Registers and updates the model's configuration for saving/sharing with
    custom attributes and auto_map for the custom model class.
    Also copies the modeling_custom.py file to the output directory so the
    model can be loaded with trust_remote_code=True.
    """
    import shutil

    HierarchicalClassifier.config_class = HierarchicalConfig
    AutoConfig.register("xlm-roberta-joke-rater", HierarchicalConfig)
    AutoModelForSequenceClassification.register(
        HierarchicalConfig, HierarchicalClassifier
    )

    current_config_dict = model.config.to_dict()
    # Instantiate the custom config using the base model's config
    model.config = HierarchicalConfig(**current_config_dict)

    n_labels = len(train_df.labels.unique())  # Should be 11 (0-10)
    model.config.num_child_labels = n_labels - 1  # 10
    model.config.class_weights_binary = binary_weights.tolist()
    model.config.class_weights_child = child_weights.tolist()

    unique_labels = sorted(train_df["labels"].unique())  # [0, 1, ..., 10]
    id2label = {int(i): int(label) for i, label in enumerate(unique_labels)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = n_labels  # Should be 11

    # Add auto_map for custom class serialization
    model.config.auto_map = {
        "AutoConfig": "modeling_custom.HierarchicalConfig",
        "AutoModelForSequenceClassification": "modeling_custom.HierarchicalClassifier",
    }

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy modeling_custom.py to output directory for trust_remote_code loading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    modeling_src = os.path.join(script_dir, "joke_rater", "modeling_custom.py")
    modeling_dst = os.path.join(output_dir, "modeling_custom.py")
    shutil.copy2(modeling_src, modeling_dst)
    print(f"Copied modeling_custom.py to {output_dir}")

    print("\nCustom config and model registered and saved successfully.")
    print(f"Generated id2label: {id2label}")


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
        from joke_rater.modeling_custom import HierarchicalClassifier

        model = HierarchicalClassifier.from_pretrained(
            MODEL_NAME,
            num_child_labels=n_labels - 1,  # 10 classes (1 to 10)
            class_weights_binary=binary_weights,
            class_weights_child=child_weights,
        )
    except ImportError:
        print(
            "FATAL ERROR: HierarchicalClassifier could not be imported. Ensure modeling_custom.py is correct."
        )
        exit(1)

    # Setup Training Arguments - configured for multi-GPU via Accelerate
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=16,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        optim="adamw_torch",
        bf16=True,  # Use bf16 for faster training on A100/H100
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        logging_steps=50,
    )

    # Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
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
        output_dir=OUTPUT_DIR,
    )

    print(f"\nFinal model and tokenizer saved to {OUTPUT_DIR}")
