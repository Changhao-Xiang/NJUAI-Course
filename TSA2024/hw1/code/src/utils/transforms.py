import numpy as np


class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data, test=False):
        return data

    def inverse_transform(self, data):
        return data


# TODO: add other transforms
class NormalizationTransform(Transform):
    def __init__(self, args) -> None:
        self.data_max = 0
        self.data_min = 0

    def transform(self, data, test=False):
        if not test:
            self.data_max = np.max(data, axis=1, keepdims=True)
            self.data_min = np.min(data, axis=1, keepdims=True)
        return (data - self.data_min) / (self.data_max - self.data_min)

    def inverse_transform(self, data):
        return data * (self.data_max - self.data_min) + self.data_min


class StandardizationTransform(Transform):
    def __init__(self, args) -> None:
        self.mean = 0
        self.std = 0

    def transform(self, data, test=False):
        if not test:
            self.mean = np.mean(data, axis=1, keepdims=True)
            var = np.sum((data - self.mean) ** 2, axis=1, keepdims=True) / data.shape[1]
            self.std = np.sqrt(var)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class MeanNormalizationTransform(Transform):
    def __init__(self, args) -> None:
        self.mean = 0
        self.data_max = 0
        self.data_min = 0

    def transform(self, data, test=False):
        if not test:
            self.mean = np.mean(data, axis=1, keepdims=True)
            self.data_max = np.max(data, axis=1, keepdims=True)
            self.data_min = np.min(data, axis=1, keepdims=True)
        return (data - self.mean) / (self.data_max - self.data_min)

    def inverse_transform(self, data):
        return data * (self.data_max - self.data_min) + self.mean


class BoxCoxTransform:
    def __init__(self, args) -> None:
        self.lamda = args.lamda

    def transform(self, data, test=False):
        if self.lamda == 0:
            return np.log(data)
        else:
            return (np.sign(data) * np.abs(data) ** self.lamda - 1) / self.lamda

    def inverse_transform(self, data):
        if self.lamda == 0:
            return np.exp(data)
        else:
            return np.sign(data * self.lamda + 1) * np.abs(data * self.lamda + 1) ** (1 / self.lamda)
