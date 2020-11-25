import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


def gen_data(N, ratio):
    """
    Generates an Nx3 dataset of (x,y,label=0,1) points sampled
    from a normal distribution with positive class mean = (-1,-1)
    and negative class mean = (1,1), with 2x2 covariance matrix
    with diagonals = 1.5
    """
    cov = np.array([[1.5, 0], [0, 1.5]])
    num_pos = int(ratio * (N / (ratio + 1)))
    num_neg = int(N / (ratio + 1))
    xpos, ypos = rnd.default_rng().multivariate_normal([-1, -1], cov, num_pos).T
    xneg, yneg = rnd.default_rng().multivariate_normal([1, 1], cov, num_neg).T

    positives = np.concatenate(([xpos], [ypos], [np.ones(num_pos)])).T
    negatives = np.concatenate(([xneg], [yneg], [np.ones(num_neg)*-1])).T
    dataset = np.concatenate((positives, negatives))
    print(dataset.shape)
    return dataset


def plot_dataset(dataset):
    labels = dataset[..., 2].ravel()
    markers = ['+' if lab == 1 else 'o' for lab in labels]
    colors = ['#1f77b4' if lab == 1 else '#ff7f0e' for lab in labels]
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], c=colors[i], marker=markers[i])
    plt.show()
