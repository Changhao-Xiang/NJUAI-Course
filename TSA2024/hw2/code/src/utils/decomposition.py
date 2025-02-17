import numpy as np


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
