"""
Upload modeling_custom.py to HuggingFace Hub models.

This is needed for trust_remote_code=True to work when loading the models.

Usage:
    python upload_custom_model_code.py KonradBRG/joke-rater-roberta-en
    python upload_custom_model_code.py --all  # Upload to all joke-rater models
"""
import argparse
import os
from huggingface_hub import HfApi, list_models


def upload_modeling_custom(repo_id: str):
    """Upload modeling_custom.py to a HuggingFace model repo."""
    api = HfApi()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    modeling_file = os.path.join(script_dir, "joke_rater", "modeling_custom.py")

    if not os.path.exists(modeling_file):
        print(f"Error: {modeling_file} not found")
        return False

    print(f"Uploading modeling_custom.py to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=modeling_file,
            path_in_repo="modeling_custom.py",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Successfully uploaded to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload modeling_custom.py to HuggingFace Hub")
    parser.add_argument(
        "repo_id",
        type=str,
        nargs="?",
        help="HuggingFace repo ID (e.g., KonradBRG/joke-rater-roberta-en)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload to all KonradBRG joke-rater models",
    )
    args = parser.parse_args()

    if args.all:
        # Find all joke-rater models by the user
        models = list_models(author="KonradBRG", search="joke-rater")
        for model in models:
            upload_modeling_custom(model.id)
    elif args.repo_id:
        upload_modeling_custom(args.repo_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
