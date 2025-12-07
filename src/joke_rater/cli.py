import argparse

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