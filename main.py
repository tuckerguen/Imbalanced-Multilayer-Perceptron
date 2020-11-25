from data_help.data_help import *
from MLP.mlp import MLP
from MLP.layer import Layer

def print_hi(name):
    # num_samples = 1000
    # # ratio:1 positive to negative class ratio
    # ratio = 19
    # dataset = gen_data(num_samples, ratio)
    # plot_dataset(dataset)
    # mlp = MLP(2, 3, np.tanh)
    l = Layer(2, 3, np.tanh)
    layer_output = l.feed_forward([0.1, 0.4])
    print(layer_output)






if __name__ == '__main__':
    print_hi('PyCharm')
