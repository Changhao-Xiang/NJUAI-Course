import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import fft
from src.models.base import MLForecastModel
from src.utils.decomposition import differential_decomposition, moving_average
from src.utils.distance import chebyshev, euclidean, manhattan
from tqdm import trange


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == "euclidean":
            self.distance = euclidean
        elif args.distance == "manhattan":
            self.distance = manhattan
        elif args.distance == "chebyshev":
            self.distance = chebyshev

        self.embedding = args.embedding
        self.decomposition = args.decomposition

        self.msas = args.msas
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.msas == "MIMO":
            if "lag" in self.embedding:
                distances = self.distance(x, X_s[:, :seq_len, :])
            elif "fft" in self.embedding:
                x_fft = fft(x)
                X_s_fft = fft(X_s[:, :seq_len, :])
                distances = self.distance(x_fft, X_s_fft)

            indices_of_smallest_k = np.argsort(distances)[: self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:, :]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore

    def _search_decompose(self, x, X_s, seq_len, pred_len):
        # Decompose input into trend and seasonal components
        if "ma" in self.decomposition:
            x_trend, x_seasonal = moving_average(x, seasonal_period=24)
            X_s_trend, X_s_seasonal = moving_average(X_s, seasonal_period=24)
        elif "differential" in self.decomposition:
            x_trend, x_seasonal = differential_decomposition(x)
            X_s_trend, X_s_seasonal = differential_decomposition(X_s)

        x_decompo = [x_trend, x_seasonal]
        X_s_decompo = [X_s_trend, X_s_seasonal]
        x_fore = np.zeros((1, pred_len, x.shape[-1]))
        for cur_x, cur_X_s in zip(x_decompo, X_s_decompo):
            if "lag" in self.embedding:
                distances = self.distance(cur_x, cur_X_s[:, :seq_len, :])
            elif "fft" in self.embedding:
                x_fft = fft(cur_x)
                X_s_fft = fft(cur_X_s[:, :seq_len, :])
                distances = self.distance(x_fft, X_s_fft)

            indices_of_smallest_k = np.argsort(distances)[: self.k]
            neighbor_fore = cur_X_s[indices_of_smallest_k, seq_len:, :]
            x_fore += np.mean(neighbor_fore, axis=0, keepdims=True)

        return x_fore

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        bs, seq_len, channels = X.shape
        X_s = sliding_window_view(self.X, (seq_len + pred_len, channels)).reshape(
            -1, seq_len + pred_len, channels
        )

        for i in trange(X.shape[0]):
            x = X[i, :, :]
            if "ma" in self.decomposition or "differential" in self.decomposition:
                x_fore = self._search_decompose(x, X_s, seq_len, pred_len)
            else:
                x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore
