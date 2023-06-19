import sys
import argparse

from config import Config
from train.datasets import get_data
from train.run_training import run_training


def _parse_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, type=str, default="gs://ml-25m")
    parser.add_argument("--loss", required=False, type=str, default="vanilla-sm")
    parser.add_argument("--nb_epochs", required=False, type=int, default=3)
    parser.add_argument("--batch_size", required=False, type=int, default=64)
    parser.add_argument("--embedding_dimension", required=False, type=int, default=32)
    return Config(vars(parser.parse_args(args=sys.argv[1:])))


if __name__ == "__main__":
    config = _parse_config()
    data = get_data(config)
    run_training(data, config)
