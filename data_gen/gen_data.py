import numpy as np
import pandas as np
import matplotlib.pyplot as plt

def gen_data(dim, num_samples, neg_proportion):
    X = np.random.normal(0, 3, (100, 1))
    Y = np.random.normal(0, 3, (100, 1))
    plt.scatter(X, Y, color='r')
    plt.show()

