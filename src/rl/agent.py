from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer # type: ignore
from scoring_model import Scorer

import weave

scorer = Scorer()

def _crowd_score_rewards(completions, **kwargs) -> list[float]:
    return scorer.crowd_score_rewards(completions, **kwargs)


def get_trainer(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", data_path: str = "data/rl_df.parquet"):
    """ Returns a GRPO trainer with the crowd scoring reward function. """
    
    dataset = load_dataset("parquet", data_files=data_path, split="train")

    training_args = GRPOConfig(output_dir=model_name + "-GRPO", report_to="wandb")
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[_crowd_score_rewards], # type: ignore
        args=training_args,
        train_dataset=dataset, # type: ignore
    )
    
    return trainer