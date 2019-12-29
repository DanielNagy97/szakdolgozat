import numpy as np

class vector():
    def __init__(self, vector):
        self.vector = vector

    def lenght(self):
        return np.sqrt(np.sum(np.power(self.vector, 2)))

    def normalize(self):
        return vector(np.divide(self.vector, self.lenght()))

    def dir_vector(self):
        return vector(np.subtract(self.vector[1], self.vector[0]))