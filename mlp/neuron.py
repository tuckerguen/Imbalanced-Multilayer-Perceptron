import numpy as np


class Neuron:
    def __init__(self, n_in, afcn):
        self.n_in = n_in
        self.weights = np.random.normal(0.0, 1.0, n_in)
        self.afcn = afcn

    def activate(self, inputs):
        return self.afcn(np.dot(self.weights, inputs))
