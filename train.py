import argparse
import yaml

from PINN.trainer import Trainer
import torch
import numpy as np
import random


class ConfigObject:
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(args_config: dict):
    args_config = ConfigObject(args_config)

    random.seed(args_config.random_state)
    np.random.seed(args_config.random_state)
    torch.manual_seed(args_config.random_state)

    trainer = Trainer(args_config)

    # TODO: add logger
    # TODO: add W&B experiment tracking
    # TODO: refactor dataset creation, create separate ds, and load it
    # TODO: validate feature scaling 
    trainer.train_model(args_config,
                D=args_config.D,
                num_epochs=args_config.num_epochs,
                alpha_boundary=args_config.alpha_boundary,
                alpha_physics=args_config.alpha_physics,
                alpha_data=args_config.alpha_data,
                learning_rate=args_config.learning_rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Complex Neural Network model.")

    parser.add_argument("--config_path", type=str,
                        help="Number of epochs for training.",
                        default="configs/training/physics_boundary_data.yaml")
    parser.add_argument("--random_state", type=int,
                    help="pytorch, numpy, python native random states",
                    default=25)
    
    args = vars(parser.parse_args())
    config = load_config(args["config_path"])
    args_config = config.copy()
    args_config.update(args)

    main(args_config)
