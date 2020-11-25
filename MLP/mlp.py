import numpy as np
from MLP.layer import Layer


class MLP:
    """
    A multilayer perceptron with one hidden layer and one output neuron
    """

    def __init__(self, n_in, n_hidden, afcn):
        self.n_in = n_in
        self.hidden = Layer(n_in, n_hidden, afcn)
        self.out_weights = np.random.normal(0.0, 1.0, n_hidden)
        self.afcn = afcn

    def predict(self, ex):
        return self.afcn(np.dot(self.out_weights, self.hidden.feed_forward(ex)))

