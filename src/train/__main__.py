from train.datasets import get_data
from train.run_training import run_training


if __name__ == "__main__":
    data = get_data(bucket_dir="gs://ml-25m")
    run_training(data)
