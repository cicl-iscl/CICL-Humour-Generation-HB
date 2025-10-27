from agent import get_trainer
from wandb import login

if __name__ == "__main__":
    login()
    trainer = get_trainer()
    trainer.train()