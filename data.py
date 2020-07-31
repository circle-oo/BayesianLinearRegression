import numpy as np


class Data:
    def __init__(self):
        self.x = []
        self.t = []
    
    def addData(self, n):
        for i in range(n):
            self.x.append(np.random.rand(1))
            self.t.append(np.sin(2*np.pi*self.x[-1]) + np.random.normal(0, 0.1))
    
    def getData(self):
        return self.x
    
    def getTarget(self):
        return self.t

