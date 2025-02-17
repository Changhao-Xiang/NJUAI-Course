import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.models import *
from src.models.base import MLForecastModel

model_dict = {
    "ZeroForecast": ZeroForecast,
    "MeanForecast": MeanForecast,
    "TsfKNN": TsfKNN,
    "ARIMA": ARIMA,
    "ThetaMethod": ThetaMethod,
    "DLinear": DLinear,
}


def get_model(args, model_name):
    return model_dict[model_name](args)


def get_base_model(args):
    return model_dict[args.base_model](args)


class ResidualModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.base_model = get_base_model(args)
        residual_models = args.residual_models.split(", ")
        self.residual_models = [get_model(args, model) for model in residual_models if model]

    def sliding_windows_to_series(self, X_windows):
        bs, seq_len, channels = X_windows.shape
        series = []
        series.extend(X_windows[0])
        for b in range(1, bs):
            series.append(X_windows[b, -1, :])
        series = np.array(series).reshape(1, -1, channels)
        return series

    def _fit(self, X: np.ndarray) -> None:
        self.base_model.fit(X)
        X_windows = np.concatenate(
            ([sliding_window_view(v, (self.seq_len + self.pred_len, v.shape[-1])) for v in X])
        )
        X_windows = X_windows.astype(np.float32)
        base_pred = self.base_model.forecast(X_windows[:, 0, : self.seq_len, :], pred_len=self.pred_len)

        residuals_windows = X_windows[:, 0, self.seq_len :, :] - base_pred.astype(np.float32)
        residuals = self.sliding_windows_to_series(residuals_windows)
        for res_model in self.residual_models:
            res_model.fit(residuals)
            residuals_windows = np.concatenate(
                ([sliding_window_view(v, (self.seq_len, v.shape[-1])) for v in residuals])
            )
            residuals_windows = residuals_windows.reshape(-1, self.seq_len, X.shape[2])

            residuals_pred = res_model.forecast(residuals_windows, pred_len=self.pred_len)

            residuals_windows = residuals_windows - residuals_pred
            residuals = self.sliding_windows_to_series(residuals_windows)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        forecasts = self.base_model.forecast(X, pred_len)

        for res_model in self.residual_models:
            forecasts += res_model.forecast(X, pred_len)

        return forecasts
