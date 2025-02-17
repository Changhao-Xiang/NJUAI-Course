from abc import abstractmethod

import numpy as np


class QAgent:
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class myQAgent(QAgent):
    def __init__(self, action_space, grid_size, lr=0.1, discount_factor=0.9) -> None:
        self.actoin_space = action_space
        self.grid_size = grid_size

        self.q_table = {}
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                self.q_table[state] = [0 for _ in range(action_space)]

        self.lr = lr
        self.discount_factor = discount_factor

    def select_action(self, obs):
        return np.argmax(self.q_table[tuple(obs)])

    def update(self, obs, action, reward, obs_next):
        max_next_q = np.max(self.q_table[tuple(obs_next)])
        current_q = self.q_table[tuple(obs)][action]

        new_q = current_q + self.lr * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[tuple(obs)][action] = new_q
