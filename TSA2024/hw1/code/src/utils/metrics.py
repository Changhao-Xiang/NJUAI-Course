import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target):
    cur_target = target.copy()
    cur_predict = predict.copy()

    zero_idx = np.where(np.isclose(target, 0))[0]
    new_target = np.delete(cur_target, zero_idx, axis=0)
    new_predict = np.delete(cur_predict, zero_idx, axis=0)
    return np.mean(np.abs((new_target - new_predict) / new_target)) * 100


def smape(predict, target):
    return np.mean(np.abs(target - predict) / (np.abs(target) + np.abs(predict))) * 200


def mase(predict, target, m=12):
    seasonal_mean_absolute_error = mae(target[m:], target[:-m])
    return np.mean(np.abs(target - predict) / seasonal_mean_absolute_error)
