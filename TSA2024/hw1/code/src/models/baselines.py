import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=-1).reshape(X.shape[0], 1)
        return np.repeat(mean, pred_len, axis=1)


# TODO: add other models based on MLForecastModel
class LinearRegressionForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.seq_len = args.seq_len

    def _fit(self, X: np.ndarray) -> None:
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

        train_data = X[:]
        subseries = np.concatenate(([sliding_window_view(v, self.seq_len + 1) for v in train_data]))
        train_X = subseries[:, : self.seq_len]
        train_Y = subseries[:, self.seq_len :]
        self.model.fit(train_X, train_Y)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        cur_X = X.copy()
        for _ in range(pred_len):
            pred = self.model.predict(cur_X)
            fore.append(pred[:, 0])
            cur_X = np.concatenate([cur_X[:, 1:], pred], axis=1)

        return np.array(fore).T


class ExponentialSmoothingForecast(MLForecastModel):
    def __init__(self, args, discount_coef=0.99) -> None:
        super().__init__()
        self.seq_len = args.seq_len
        self.discount_coef = discount_coef

        self.discount_array = np.array([self.discount_coef**i for i in range(self.seq_len)])
        self.scale_factor = (1 - self.discount_coef) / (1 - self.discount_coef**self.seq_len)

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        cur_X = self.scale_factor * np.sum(X * self.discount_array, axis=1).reshape(-1, 1)
        fore = [cur_X[:, 0]]

        pred = (1 - self.discount_coef) * X[-1] + self.discount_coef * cur_X
        for _ in range(pred_len - 1):
            fore.append(pred[:, 0])

        return np.array(fore).T
