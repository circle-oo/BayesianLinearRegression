from typing import Tuple

import numpy as np


class Data:
    def __init__(self, mean: float = .0, std: float = .01):
        self.mean = mean
        self.std = std

        self.x = np.empty((0, 1))
        self.y = np.empty((0, 1))
    
    def add(self, n: int):
        x = np.random.rand(n, 1)
        y = np.sin(2 * np.pi * x) + np.random.normal(self.mean, self.std, size=(n, 1))

        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
    
    def get(self) \
            -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y
