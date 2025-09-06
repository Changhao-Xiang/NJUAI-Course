import argparse
import os

import numpy as np
from framework import nn
from framework.data import CifarDataset, DataLoader, MNISTDataset
from framework.optim import SGD, Adam
from tqdm import tqdm


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main_path = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())


def ResMLP(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    resmlp = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes),
    )
    return resmlp


def MLP(dim, hidden_dim, num_classes, layers, drop_prob=0.1):
    hidden_layer_factory = lambda: [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(drop_prob)]
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        *sum([hidden_layer_factory() for _ in range(layers - 2)], start=[]),
        nn.Linear(hidden_dim, num_classes),
    )


def train_eval_epoch(dataloader, model, opt=None):
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in dataloader:
            X = X.reshape((X.shape[0], -1))
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in dataloader:
            X = X.reshape((X.shape[0], -1))
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error / sample_nums, np.mean(tot_loss)


def run_mnist(
    args: argparse.Namespace,
    batch_size=64,
    epochs=10,
    optimizer=SGD,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=64,
):
    data_dir = "data/MNIST"
    resmlp = ResMLP(28 * 28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resmlp.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_save_path = "ckpt/mnist_resmlp_weights.npz"
    if not args.test_only:
        for _ in tqdm(range(epochs)):
            train_err, train_loss = train_eval_epoch(train_loader, resmlp, opt)
            tqdm.write(f"Training Error: {train_err}, Training Loss: {train_loss}")

        test_err, test_loss = train_eval_epoch(test_loader, resmlp, None)
        tqdm.write(f"Test Error: {test_err}, Test Loss: {test_loss}")

        # Save model weights
        resmlp.save_weights(model_save_path)

    else:
        # Load model weights
        resmlp.load_weights(model_save_path)
        test_err, test_loss = train_eval_epoch(test_loader, resmlp, None)
        tqdm.write(f"Test Error: {test_err}, Test Loss: {test_loss}")


def run_cifar(
    args: argparse.Namespace,
    batch_size=64,
    epochs=10,
    optimizer=SGD,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=64,
):
    resmlp = ResMLP(32 * 32 * 3, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resmlp.parameters(), lr=lr, weight_decay=weight_decay)

    train_files = [os.path.join("data/cifar10", f) for f in os.listdir("data/cifar10") if f.startswith('data_batch_')]
    test_file = os.path.join("data/cifar10", 'test_batch')
    train_set = CifarDataset(train_files)
    test_set = CifarDataset([test_file])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_save_path = "ckpt/cifar_resmlp_weights.npz"
    if not args.test_only:
        for _ in tqdm(range(epochs)):
            train_err, train_loss = train_eval_epoch(train_loader, resmlp, opt)
            tqdm.write(f"Training Error: {train_err}, Training Loss: {train_loss}")

        test_err, test_loss = train_eval_epoch(test_loader, resmlp, None)
        tqdm.write(f"Test Error: {test_err}, Test Loss: {test_loss}")

        # Save model weights
        resmlp.save_weights(model_save_path)

    else:
        # Load model weights
        resmlp.load_weights(model_save_path)
        test_err, test_loss = train_eval_epoch(test_loader, resmlp, None)
        tqdm.write(f"Test Error: {test_err}, Test Loss: {test_loss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--test_only", default=False, action="store_true")
    args = parser.parse_args()
    if args.dataset == "mnist":
        run_mnist(args)
    elif args.dataset == "cifar":
        run_cifar(args)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
