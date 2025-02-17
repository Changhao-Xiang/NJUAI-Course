import numpy as np


def euclidean(A, B):
    return np.sqrt(np.sum(np.sum((A - B) ** 2, axis=1), axis=1))


# TODO: add more distance
def manhattan(A, B):
    return np.sum(np.sum(np.abs(A - B), axis=1), axis=1)


def chebyshev(A, B):
    return np.max(np.max(np.abs(A - B), axis=1), axis=1)
