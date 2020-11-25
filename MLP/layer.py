import numpy as np
from MLP.neuron import Neuron

class Layer:
    """
    A hidden layer in the multilayer perceptron
    """

    def __init__(self, n_in, n, afcn):
        self.n = n
        # Weights from input nodes to layer nodes
        self.neurons = [Neuron(n_in, afcn)] * n
        # Node biases
        self.biases = np.zeros(n)

    def feed_forward(self, inputs):
        """
        Compute outputs of all neurons in layer
        """
        return [n.activate(inputs) for n in self.neurons]

    def back_prop(self, loss):
        pass

