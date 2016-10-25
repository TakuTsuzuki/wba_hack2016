import numpy as np

class Grass(object):
    def __init__(self, size=100, threthold=0.1):
        self.size = size
        self.threthold = threthold
        self.world = (np.random.rand(self.size,self.size) < self.threthold).astype(np.float32)
    
    def process(self):
        kernel = np.random.rand(self.size,self.size) < self.threthold
        self.world = np.logical_or(self.world, kernel).astype(np.float32)
        return self.world