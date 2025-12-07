import argparse

def parse_args():
    """Parses command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a Joke Generation Policy using GRPO.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID for the policy model (LLM) to be trained."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Llama-3.2-1B-Instruct-GRPO",
        help="Directory to save the trained GRPO model and logs."
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="data/rl_df_train.parquet",
        help="Path to the training dataset (.parquet file)."
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default="data/rl_df_test.parquet",
        help="Path to the evaluation dataset (.parquet file)."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for the GRPO optimization."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Whether to use vLLM for efficient generation."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Reporting tool to use (e.g., 'wandb', 'none')."
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=64,
        help="Maximum length of the generated joke completions."
    )
    
    # Batch/Generation settings
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per device for evaluation.")
    parser.add_argument("--generation_batch_size", type=int, default=16, help="Batch size for parallel generation.")
    parser.add_argument("--num_generations", type=int, default=16, help="Number of responses to generate per prompt (G in GRPO).")

    return parser.parse_args()
