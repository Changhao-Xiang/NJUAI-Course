import argparse
import os

import numpy as np
from framework import nn
from framework.data import CifarDataset, DataLoader, MNISTDataset
from framework.optim import SGD, Adam
from tqdm import tqdm


def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Basic convolutional block with Conv2d + BatchNorm + ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU()
    )


def ResNetBlock(in_channels, out_channels, stride=1):
    """ResNet basic block"""
    # Main path
    main_path = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    )
    
    # Skip connection
    if stride != 1 or in_channels != out_channels:
        # Need to adjust dimensions
        skip_path = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        
        class ResBlockWithProjection(nn.Module):
            def __init__(self, main_path, skip_path):
                super().__init__()
                self.main_path = main_path
                self.skip_path = skip_path
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return self.relu(self.main_path(x) + self.skip_path(x))
        
        return ResBlockWithProjection(main_path, skip_path)
    else:
        # Identity skip connection
        class ResBlockIdentity(nn.Module):
            def __init__(self, main_path):
                super().__init__()
                self.main_path = main_path
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return self.relu(self.main_path(x) + x)
        
        return ResBlockIdentity(main_path)


def SimpleResNet(num_classes=10):
    """Simple ResNet for CIFAR-10"""
    return nn.Sequential(
        # Initial conv layer
        nn.Conv2d(3, 8, 3, 1, 1),  # 32x32x3 -> 32x32x32
        nn.ReLU(),
        
        # residual block with downsampling
        ResNetBlock(8, 8, 2),     # 32x32x8 -> 16x16x8
        ResNetBlock(8, 8, 1),     # 16x16x8 -> 16x16x8
        
        # Global average pooling and classifier
        nn.AdaptiveAvgPool2d((1, 1)),  # 16x16x8 -> 1x1x8
        nn.Flatten(),                   # 1x1x8 -> 8
        nn.Linear(8, num_classes)     # 8 -> 10
    )


def train_eval_epoch(dataloader, model, opt=None):
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error / sample_nums, np.mean(tot_loss)

def run_cifar(
    args: argparse.Namespace,
    batch_size=16,
    epochs=10,
    optimizer=Adam,
    lr=0.001,
    weight_decay=0.0001,
):
    model = SimpleResNet(num_classes=10)
    model_save_path = "ckpt/cifar_resnet_weights.npz"
    
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_files = [os.path.join("data/cifar10", f) for f in os.listdir("data/cifar10") if f.startswith('data_batch_')]
    test_file = os.path.join("data/cifar10", 'test_batch')
    train_set = CifarDataset(train_files, num_samples=512)
    test_set = CifarDataset([test_file], num_samples=128)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if not args.test_only:
        for epoch in tqdm(range(epochs)):
            train_err, train_loss = train_eval_epoch(train_loader, model, opt)
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Training Error: {train_err:.4f}, Training Loss: {train_loss:.4f}")

        test_err, test_loss = train_eval_epoch(test_loader, model, None)
        tqdm.write(f"Test Error: {test_err:.4f}, Test Loss: {test_loss:.4f}")

        # Save model weights
        model.save_weights(model_save_path)

    else:
        # Load model weights
        model.load_weights(model_save_path)
        test_err, test_loss = train_eval_epoch(test_loader, model, None)
        tqdm.write(f"Test Error: {test_err:.4f}, Test Loss: {test_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar", choices=["mnist", "cifar"])
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn"], 
                       help="Model type: mlp for ResMLP, cnn for CNN-ResNet")
    args = parser.parse_args()
    run_cifar(args)


if __name__ == "__main__":
    main()
