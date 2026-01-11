"""
Train a language-specific RoBERTa joke rater model.

Usage:
    python train_roberta.py --language en --output_dir ./roberta-en
    python train_roberta.py --language zh --output_dir ./roberta-zh
    python train_roberta.py --language es --output_dir ./roberta-es
"""
import os
import shutil
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error, accuracy_score, mean_absolute_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import HfApi
from joke_rater.cli import parse_args
from joke_rater.preprocessing import load_datasets
from joke_rater.modeling_custom import HierarchicalConfig, HierarchicalClassifier


def register_custom_model():
    """Register the custom model classes with transformers AutoModel."""
    HierarchicalClassifier.config_class = HierarchicalConfig
    AutoConfig.register(
        HierarchicalConfig.model_type,
        HierarchicalConfig,
        exist_ok=True,
    )
    AutoModelForSequenceClassification.register(
        HierarchicalConfig,
        HierarchicalClassifier,
        exist_ok=True,
    )


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

    Returns Python lists (JSON serializable) for config storage.
    The model will convert them to tensors internally.
    """
    labels = np.array(train_ds["labels"])

    # 1. Binary Weights (0 vs. 1-10)
    binary_counts = np.array([(labels == 0).sum(), (labels != 0).sum()], dtype=float)
    binary_weights = binary_counts.sum() / (2 * binary_counts)

    # 2. Child Weights (1 to 10, only for non-zero labels)
    child_labels = labels[labels != 0]
    num_child_classes = 10  # Labels 1 through 10
    child_counts = np.array(
        [(child_labels == c).sum() for c in range(1, num_child_classes + 1)],
        dtype=float,
    )

    # Handle zero counts to avoid division by zero
    child_counts = np.maximum(child_counts, 1)
    child_weights = child_counts.sum() / (num_child_classes * child_counts)

    # Return as Python lists (JSON serializable)
    return binary_weights.tolist(), child_weights.tolist()


def setup_model_config(model, train_df, binary_weights, child_weights):
    """
    Update the model's configuration with custom attributes needed for inference.
    This sets up id2label, label2id, and auto_map for trust_remote_code loading.
    """
    n_labels = len(train_df.labels.unique())  # Should be 11 (0-10)

    # Convert config to HierarchicalConfig if needed
    current_config_dict = model.config.to_dict()
    model.config = HierarchicalConfig(**current_config_dict)

    model.config.num_child_labels = n_labels - 1  # 10
    model.config.class_weights_binary = binary_weights
    model.config.class_weights_child = child_weights

    # Set up label mappings - labels are 0-10, indices are 0-10
    unique_labels = sorted(train_df["labels"].unique())  # [0, 1, ..., 10]
    id2label = {int(i): int(label) for i, label in enumerate(unique_labels)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = n_labels  # 11

    # Add auto_map for custom class serialization with trust_remote_code
    model.config.auto_map = {
        "AutoConfig": "modeling_custom.HierarchicalConfig",
        "AutoModelForSequenceClassification": "modeling_custom.HierarchicalClassifier",
    }

    print(f"Configured model with {n_labels} labels")
    print(f"id2label: {id2label}")

    return model


def save_and_push_to_hub(model, tokenizer, output_dir, hub_model_id):
    """
    Save model, tokenizer, and modeling_custom.py locally and push everything to Hub.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model and tokenizer locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # Copy modeling_custom.py to output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    modeling_src = os.path.join(script_dir, "joke_rater", "modeling_custom.py")
    modeling_dst = os.path.join(output_dir, "modeling_custom.py")
    shutil.copy2(modeling_src, modeling_dst)
    print(f"Copied modeling_custom.py to {output_dir}")

    # Push everything to Hub
    print(f"\nPushing to Hub: {hub_model_id}")
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=hub_model_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the entire output directory (model, tokenizer, modeling_custom.py)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=hub_model_id,
        repo_type="model",
    )
    print(f"Successfully pushed all files to {hub_model_id}")


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
    LANGUAGE = args.language
    DATA_DIR = args.data_dir

    print(f"--- Starting Training ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Hub Model ID: {HUB_MODEL_ID}")
    print(f"Data Directory: {DATA_DIR}")

    # Register custom model classes
    register_custom_model()

    # Data Loading and Preprocessing (language-specific)
    datasets, tokenizer, train_df = load_datasets(
        MODEL_NAME, language=LANGUAGE, data_dir=DATA_DIR
    )
    train_ds = datasets["train"]
    val_ds = datasets["validation"]
    test_ds = datasets["test"]

    # Calculate Class Weights
    binary_weights, child_weights = get_cross_entropy_weights(train_ds)
    n_labels = len(train_df.labels.unique())

    # Initialize Custom Model
    model = HierarchicalClassifier.from_pretrained(
        MODEL_NAME,
        num_child_labels=n_labels - 1,
        class_weights_binary=binary_weights,
        class_weights_child=child_weights,
    )

    # Setup Training Arguments
    # Note: push_to_hub=False - we push manually after training with all files
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
        push_to_hub=False,  # We push manually after training
        optim="adamw_torch",
        bf16=True,
        dataloader_num_workers=4,
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

    print("\n--- Starting Training ---")
    trainer.train()

    # Evaluate on Test Set
    test_results = trainer.predict(test_ds)
    print("\n--- Test Results ---")
    print(test_results.metrics)

    # Get the best model (loaded automatically due to load_best_model_at_end=True)
    best_model = trainer.model

    # Setup the model config with proper id2label, label2id, and auto_map
    best_model = setup_model_config(
        model=best_model,
        train_df=train_df,
        binary_weights=binary_weights,
        child_weights=child_weights,
    )

    # Save locally and push to Hub (model + tokenizer + modeling_custom.py)
    save_and_push_to_hub(
        model=best_model,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        hub_model_id=HUB_MODEL_ID,
    )

    print(f"\n--- Training Complete ---")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")
