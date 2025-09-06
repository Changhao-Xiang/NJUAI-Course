import gzip
import pickle
import struct
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .autograd import Tensor


class Dataset(ABC):
    def __init__(self, transforms: Optional[list] = None):
        self.transforms = transforms
        self.features: np.ndarray
        self.labels: np.ndarray

    @abstractmethod
    def __getitem__(self, index) -> object:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.idx : end_idx]
        batch = [self.dataset[i] for i in batch_indices]
        batch_features = np.stack([x[0] for x in batch])
        batch_labels = np.array([x[1] for x in batch])
        self.idx += self.batch_size

        return Tensor(batch_features), Tensor(batch_labels)


class MNISTDataset(Dataset):
    def __init__(self, image_filename: str, label_filename: str, transforms: Optional[list] = None):
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.features = (
                np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols, 1).astype(np.float32)
            )
            self.features /= 255.0

        with gzip.open(label_filename, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        img = self.features[index]
        if self.transforms:
            img = self.apply_transforms(img)
        return img, self.labels[index]


class CifarDataset(Dataset):
    def __init__(self, files: list[str], transforms: Optional[list] = None, num_samples: int = 128):
        super().__init__(transforms)

        features = []
        labels = []

        for file in files:
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            
            # CIFAR-10 pickle files typically have keys: b'data', b'labels' or b'fine_labels'
            # Data is stored as (N, 3072) array where each row is a 32x32x3 image flattened
            if b'data' in data_dict:
                # Standard CIFAR-10 format
                cur_features = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
                cur_features /= 255.0  # Normalize to [0, 1]
                features.append(cur_features)
                # Labels can be under different keys
                if b'labels' in data_dict:
                    cur_labels = np.array(data_dict[b'labels'])
                elif b'fine_labels' in data_dict:
                    cur_labels = np.array(data_dict[b'fine_labels'])
                else:
                    raise ValueError("No labels found in CIFAR data file")
                labels.append(cur_labels)
            else:
                raise ValueError("Invalid CIFAR data file format - 'data' key not found")

        self.features = np.concatenate(features)[:num_samples]
        self.labels = np.concatenate(labels)[:num_samples]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        img = self.features[index]
        if self.transforms:
            img = self.apply_transforms(img)
        return img, self.labels[index]
