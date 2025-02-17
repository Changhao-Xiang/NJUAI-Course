import argparse
import random

import numpy as np
from src.dataset.data_visualizer import data_visualize
from src.dataset.dataset import get_dataset
from src.models.baselines import MeanForecast, ZeroForecast
from src.models.TsfKNN import TsfKNN
from src.utils.transforms import IdentityTransform
from trainer import MLTrainer


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument("--data_path", type=str, default="./dataset/electricity/electricity.csv")
    parser.add_argument(
        "--dataset", type=str, default="Custom", help="dataset type, options: [M4, ETT, Custom]"
    )
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument("--ratio_train", type=int, default=0.7, help="train dataset length")
    parser.add_argument("--ratio_val", type=int, default=0, help="validate dataset length")
    parser.add_argument("--ratio_test", type=int, default=0.3, help="input sequence length")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    args = get_args()

    dataset = get_dataset(args)
    data_visualize(dataset, t=96)
