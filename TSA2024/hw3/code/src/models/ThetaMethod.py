import numpy as np
from src.models.base import MLForecastModel
from statsmodels.tsa.forecasting.theta import ThetaModel


class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.models = []
        for c in range(X.shape[2]):  # Fit a Theta model for each channel
            model = ThetaModel(X[0, :, c], period=X.shape[1])
            self.models.append(model.fit())

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        bs, seq_len, channels = X.shape
        total_pred_len = bs + seq_len + pred_len - 1

        forecasts = []
        for model in self.models:
            forecast = model.forecast(steps=total_pred_len)
            forecasts.append(forecast)
        forecasts = np.stack(forecasts, axis=-1)

        # Convert to sliding windows shape
        forecasts_windows = np.zeros((bs, pred_len, channels))
        for b in range(bs):
            forecasts_windows[b, :, :] = forecasts[b + seq_len : b + seq_len + pred_len, :]
        return forecasts_windows
