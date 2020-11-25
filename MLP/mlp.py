import numpy as np


class MLP:
    """
    A multilayer perceptron with one hidden layer and one output neuron
    """

    def __init__(self, n_in, n_hidden, afcn):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.afcn = afcn

