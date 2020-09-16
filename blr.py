from functools import partial

import numpy as np
import matplotlib.pyplot as plt


class BLR:
    def __init__(self, alpha: float, beta: float, dimension: int):
        self.a = alpha
        self.b = beta
        self.D = dimension

        self.m = np.zeros(self.D, dtype=np.float)
        self.S = np.zeros((self.D, self.D), dtype=np.float)

    def f(self, x: np.ndarray, j: int):
        return pow(x, j)

    def fit(self, data: np.ndarray, target: np.ndarray):
        self.P = np.zeros((self.D+1, len(data)), dtype=np.float)
        for i in range(self.D + 1):
            for j in range(len(data)):
                self.P[i][j] = self.f(data[j], i)

        self.S = np.linalg.inv(self.a * np.identity(self.D + 1) + self.b * np.matmul(self.P, np.transpose(self.P))) #(D, D)
        self.m = self.b * np.matmul(np.matmul(self.S, self.P), target.reshape(len(target), 1))
        

    def predict(self, x: np.ndarray, data: np.ndarray, target: np.ndarray) \
            -> np.ndarray:

        self.w = np.random.multivariate_normal(np.resize(self.m, len(self.m)), self.S)
        ww = np.array(tuple(map(partial(self.f, x), range(self.D + 1))))
        
        return np.dot(self.w, ww)
    
    def plot(self, data: np.ndarray, target: np.ndarray, n: int):
        x = np.arange(0, 1, 0.05)

        plt.plot(data, target, 'ob')
        for _ in range(n):
            pred = self.predict(x, data, target)
            plt.plot(x, pred, 'r')
        plt.show()
