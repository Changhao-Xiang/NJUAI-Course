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

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


# TODO: add other transforms
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
