import numpy as np
from MLP.neuron import Neuron


class Layer:
    """
    A hidden layer in the multilayer perceptron
    """

    def __init__(self, n_in, n, afcn):
        self.n = n
        # Weights from input nodes to layer nodes
        self.weights = np.random.normal(0.0, np.sqrt(2 / (n_in + n)), (n_in, n))
        # Node biases
        self.biases = np.zeros(n)

    def feed_forward(self, inputs):
        """
        Compute outputs of all neurons in layer
        """
        return np.dot(inputs, self.weights) + self.biases

    def back_prop(self, loss):
        pass
