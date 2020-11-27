import numpy as np
from MLP.neuron import Neuron


class Layer:
    """
    A hidden layer in the multilayer perceptron
    """

    def __init__(self, n_in, n_out, afcn):
        self.n = n_out
        # Weights from input nodes to layer nodes
        self.weights = np.random.normal(0.0, np.sqrt(2 / (n_in + n_out)), (n_in, n_out))
        # Node biases
        self.biases = np.zeros(n_out)
        # Activation function
        self.afcn = afcn
        # The inputs for each example across all examples
        self.ex_inputs = None
        # The sum of all inputs to each node for each example over all examples
        self.ex_sums = None

    def feed_forward(self, inputs):
        """
        Computes output of all neurons in this layer
        :param inputs: input to the layer
        :return: array of outputs, one element for each neuron
        """
        # The most recent set of inputs
        if self.ex_inputs is None:
            self.ex_inputs = np.array(inputs)
        else:
            self.ex_inputs = np.vstack((self.ex_inputs, inputs))
        sums = np.dot(inputs, self.weights)
        if self.ex_sums is None:
            self.ex_sums = sums
        else:
            self.ex_sums = np.vstack((self.ex_sums, sums))
        return self.afcn(sums + self.biases)

