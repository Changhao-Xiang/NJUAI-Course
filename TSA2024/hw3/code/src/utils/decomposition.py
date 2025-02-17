import numpy as np
from statsmodels.tsa.seasonal import STL


def moving_average(x, seasonal_period):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    if x.ndim == 2:
        x = x.reshape(-1, x.shape[0], x.shape[1])
    assert x.ndim == 3

    trend = np.zeros_like(x)
    bs, seq_len, channels = x.shape

    for b in range(bs):
        for c in range(channels):
            trend[b, :, c] = np.convolve(x[b, :, c], np.ones(seasonal_period) / seasonal_period, mode="same")

    seasonal = x - trend
    return trend, seasonal


def differential_decomposition(x):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    if x.ndim == 2:
        x = x.reshape(-1, x.shape[0], x.shape[1])
    assert x.ndim == 3

    differences = np.diff(x, axis=1)
    differences = np.concatenate([differences, x[:, 0, :].reshape(-1, 1, x.shape[-1])], axis=1)
    trend = np.cumsum(differences, axis=1)
    seasonal = x - trend

    return trend, seasonal


def STL_decomposition(x, seasonal_period):
    """
    Seasonal and Trend decomposition using Loess
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        residual (numpy.ndarray): Residual component
    """
    if x.ndim == 2:
        x = x.reshape(-1, x.shape[0], x.shape[1])
    assert x.ndim == 3

    bs, seq_len, channels = x.shape
    trend = np.zeros_like(x)
    seasonal = np.zeros_like(x)
    residual = np.zeros_like(x)

    for b in range(bs):
        for c in range(channels):
            stl = STL(x[b, :, c], period=seasonal_period)
            result = stl.fit()
            trend[b, :, c] = result.trend
            seasonal[b, :, c] = result.seasonal
            residual[b, :, c] = result.resid

    return trend, seasonal, residual


def X11_decomposition(x, seasonal_period):
    """
    X11 decomposition
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        residual (numpy.ndarray): Residual component
    """
    if x.ndim == 2:
        x = x.reshape(-1, x.shape[0], x.shape[1])
    assert x.ndim == 3
    bs, seq_len, channels = x.shape

    trend = np.zeros_like(x)
    seasonal = np.zeros_like(x)
    residual = np.zeros_like(x)

    for b in range(bs):
        for c in range(channels):
            # Extract the current channel data
            current_channel = x[b, :, c]

            # Initial trend estimate using 2x12 moving average
            trend_channel = np.zeros(seq_len)
            for i in range(seq_len):
                if i < seasonal_period or i >= seq_len - seasonal_period:
                    trend_channel[i] = np.nan
                else:
                    trend_channel[i] = np.mean(current_channel[i - seasonal_period : i + seasonal_period + 1])

            # Remove trend to get seasonal-irregular component
            seasonal_irregular = current_channel / trend_channel

            # Calculate seasonal indices
            seasonal_indices = np.zeros(seasonal_period)
            for j in range(seasonal_period):
                seasonal_indices[j] = np.nanmean(seasonal_irregular[j::seasonal_period])

            # Normalize seasonal indices
            seasonal_indices /= np.mean(seasonal_indices)

            # Remove seasonal component to get trend-irregular component
            trend_irregular = current_channel / seasonal_indices[np.arange(seq_len) % seasonal_period]

            # Refine trend estimate using Henderson moving average
            trend_channel = np.zeros(seq_len)
            for i in range(seq_len):
                if i < 6 or i >= seq_len - 6:
                    trend_channel[i] = np.nan
                else:
                    trend_channel[i] = np.mean(trend_irregular[i - 6 : i + 7])

            # Remove refined trend to get residual
            residual_channel = trend_irregular / trend_channel

            # Remove residual to get final seasonal component
            seasonal_channel = current_channel / trend_irregular

            # Store the results for the current channel
            trend[b, :, c] = trend_channel
            seasonal[b, :, c] = seasonal_channel
            residual[b, :, c] = residual_channel

    return trend, seasonal, residual
