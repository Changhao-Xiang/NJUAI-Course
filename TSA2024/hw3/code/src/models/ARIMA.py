import numpy as np
from src.models.base import MLForecastModel
from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
from statsmodels.tsa.stattools import acf, adfuller, pacf


class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _determine_order(self, X: np.ndarray):
        # Determine the order of differencing (d) using ADF test
        def adf_test(series):
            result = adfuller(series)
            return result[1] <= 0.05  # If p-value is less than 0.05, the series is stationary

        d = 0
        while not adf_test(X[:, :, 0].flatten()) and d < 3:
            X = np.diff(X, axis=1)
            d += 1

        # Determine p and q using ACF and PACF
        acf_vals = acf(X[:, :, 0].flatten(), nlags=20)
        pacf_vals = pacf(X[:, :, 0].flatten(), nlags=20)

        p = np.argmax(pacf_vals > 2 / np.sqrt(len(X)))
        q = np.argmax(acf_vals > 2 / np.sqrt(len(X)))

        return p, d, q

    def _fit(self, X: np.ndarray) -> None:
        self.models = []
        order = self._determine_order(X)

        for c in range(X.shape[2]):  # Fit an ARIMA model for each channel
            model = ARIMA_Model(X[0, :, c], order=order)
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
