from agent import get_trainer
from get_data import prepare_rl_data
from wandb import login

if __name__ == "__main__":
    login()
    prepare_rl_data()
    trainer = get_trainer()
    trainer.train()