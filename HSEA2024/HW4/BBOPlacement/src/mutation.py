import numpy as np


class SwapMutation:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        new_x = x.copy()

        # Randomly select two different macros to swap
        macro_num = len(x) // 2
        idx1, idx2 = np.random.choice(macro_num, 2, replace=False)

        # Swap x coordinates
        new_x[2 * idx1], new_x[2 * idx2] = new_x[2 * idx2], new_x[2 * idx1]
        # Swap y coordinates
        new_x[2 * idx1 + 1], new_x[2 * idx2 + 1] = new_x[2 * idx2 + 1], new_x[2 * idx1 + 1]

        return new_x


class ResetMutation:
    def __init__(self, lb, ub) -> None:
        self.lb = np.max(lb)
        self.ub = np.min(ub)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        new_x = x.copy()

        # Randomly reset each macro's position independently
        macro_num = len(x) // 2
        for idx in range(macro_num):
            if np.random.rand() < 0.01:
                new_x[2 * idx] = np.random.randint(self.lb, self.ub + 1)
                new_x[2 * idx + 1] = np.random.randint(self.lb, self.ub + 1)

        return new_x


class CreepMutation:
    def __init__(self, lb, ub) -> None:
        self.lb = np.max(lb)
        self.ub = np.min(ub)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        new_x = x.copy()

        new_x += np.random.randint(-1, 2, size=x.shape)
        new_x = np.clip(new_x, self.lb, self.ub)
        return new_x
