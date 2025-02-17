import argparse
import random

import numpy as np
from src.dataset.dataset import get_dataset
from src.models.baselines import MeanForecast, ZeroForecast
from src.models.DLinear import DLinear
from src.models.TsfKNN import TsfKNN
from src.utils.transforms import IdentityTransform, StandardizationTransform
from trainer import MLTrainer


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument("--data_path", type=str, default="./dataset/ETT/ETTh1.csv")
    # parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    # parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument("--dataset", type=str, default="ETT", help="dataset type, options: [M4, ETT, Custom]")
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument("--ratio_train", type=int, default=0.7, help="train dataset length")
    parser.add_argument("--ratio_val", type=int, default=0, help="validate dataset length")
    parser.add_argument("--ratio_test", type=int, default=0.3, help="input sequence length")
    parser.add_argument(
        "--frequency", type=str, default="h", help="frequency of time series data, options: [h, m]"
    )

    # forcast task config
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=32, help="prediction sequence length")
    parser.add_argument("--enc_in", type=int, default=7, help="input channels")

    # model define
    parser.add_argument("--model", type=str, required=False, default="TsfKNN", help="model name")
    parser.add_argument("--n_neighbors", type=int, default=3, help="number of neighbors used in TsfKNN")
    parser.add_argument("--distance", type=str, default="euclidean", help="distance used in TsfKNN")
    parser.add_argument(
        "--msas",
        type=str,
        default="MIMO",
        help="multi-step ahead strategy used in TsfKNN, options: " "[MIMO, recursive]",
    )
    parser.add_argument("--embedding", type=str, default="lag")

    # transform define
    parser.add_argument("--transform", type=str, default="StandardizationTransform")

    # decomposition define
    parser.add_argument(
        "--decomposition", type=str, default="differential", help="options: [ma, differential, no]"
    )

    args = parser.parse_args()
    return args


def get_model(args):
    model_dict = {
        "ZeroForecast": ZeroForecast,
        "MeanForecast": MeanForecast,
        "TsfKNN": TsfKNN,
        "DLinear": DLinear,
    }
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        "IdentityTransform": IdentityTransform,
        "StandardizationTransform": StandardizationTransform,
    }
    return transform_dict[args.transform](args)


if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    args = get_args()
    # load dataset
    dataset = get_dataset(args)
    # create model
    model = get_model(args)
    # data transform
    transform = get_transform(args)
    # create trainer
    trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
    # train model
    trainer.train()
    # evaluate model
    trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
