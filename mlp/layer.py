import numpy as np
from mlp.neuron import Neuron


class Layer:
    """
    A hidden layer in the multilayer perceptron
    """

    def __init__(self, n_in, n_out, afcn):
        self.n_out = n_out
        self.n_in = n_in
        # Weights from input nodes to layer nodes
        self.weights = np.random.normal(0.0, np.sqrt(2 / (n_in + n_out)), (n_in, n_out))
        # self.weights = np.ones((n_in, n_out)) * 0.0001
        if self.weights.shape[1] == 1:
            self.weights = self.weights.flatten('C')
        # Node biases
        self.biases = np.zeros(n_out)
        # Activation function
        self.afcn = afcn
        # The inputs for each example across all examples
        self.ex_inputs = None
        # The sum of all inputs to each node for each example over all examples
        self.ex_sums = None

    def randomize_weights(self):
        self.weights = np.random.normal(0.0, np.sqrt(2 / (self.n_in + self.n_out)), (self.n_in, self.n_out))

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

