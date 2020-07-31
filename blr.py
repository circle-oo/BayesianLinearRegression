import numpy as np
import matplotlib.pyplot as plt

#data generate: sin(2pix)
#BLR


class BLR:
    def __init__(self, alpha, beta, dimension):
        self.a = alpha
        self.b = beta
        self.D = dimension
        self.m = np.zeros(self.D, dtype='float32')
        self.S = np.zeros((self.D, self.D), dtype='float32')

    def f(self, x, j):
        return pow(x, j)

    def getP(self, data):
        self.P = np.zeros((self.D+1, len(data)), dtype='float32')
        for i in range(self.D + 1):
            for j in range(len(data)):
                self.P[i][j] = self.f(data[j], i)
    
    def fit(self, data, target):
        self.getP(data)
        self.S = np.linalg.inv(self.a * np.identity(self.D + 1) + self.b * np.matmul(self.P, np.transpose(self.P))) #(D, D)
        self.m = self.b * np.matmul(np.matmul(self.S, self.P), np.resize(target, (len(target),1)))

    def predict(self, data, target):
        self.w = np.random.multivariate_normal(np.resize(self.m, len(self.m)), self.S)
    
    def g(self, x):
        return np.dot(self.w, np.array([self.f(x, i) for i in range(self.D+1)]))

    def getImage(self, data, target, n):
        x = np.arange(0, 1, 0.05)
        plt.plot(data, target, 'ob')
        for t in range(n):
            self.predict(data, target)
            plt.plot(x, self.g(x), 'r')
        plt.show()
