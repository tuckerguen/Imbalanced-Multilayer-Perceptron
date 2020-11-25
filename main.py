from data_help.data_help import *
from MLP.mlp import MLP
from MLP.layer import Layer
import time

def print_hi(name):
    num_samples = 1000
    # ratio:1 positive to negative class ratio
    ratio = 19
    dataset = gen_data(num_samples, ratio)
    # plot_dataset(dataset)
    mlp = MLP(2, 3, np.tanh)
    start_time = time.time()
    prediction = mlp.predict([0.1, 0.4])
    print("--- %s seconds ---" % (time.time() - start_time))
    print(prediction)


if __name__ == '__main__':
    print_hi('PyCharm')
