import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from data_help.exset_ops import get_data, map_nominal_attr


def gen_data(N, num_attr, ratio):
    """
    Generates an Nx3 dataset of (x,y,label=0,1) points sampled
    from a normal distribution with positive class mean = (-1,-1)
    and negative class mean = (1,1), with 2x2 covariance matrix
    with diagonals = 1.5
    """
    cov = np.identity(num_attr)
    num_pos = int(ratio * (N / (ratio + 1)))
    num_neg = int(N / (ratio + 1))
    allpos = rnd.default_rng().multivariate_normal([-1] * num_attr, cov, num_pos).T
    allneg = rnd.default_rng().multivariate_normal([1] * num_attr, cov, num_neg).T

    positives = allpos.T
    negatives = allneg.T
    dataset = np.concatenate((np.concatenate((allpos, [np.ones(num_pos)])).T,
                              np.concatenate((allneg, [np.ones(num_neg) * -1])).T))
    return dataset, positives, negatives


def plot_dataset(dataset):
    print("Plotting dataset")
    labels = dataset[..., 2].ravel()
    markers = ['+' if lab > 0 else 'o' for lab in labels]
    colors = ['#1f77b4' if lab > 0 else '#ff7f0e' for lab in labels]
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], c=colors[i], marker=markers[i], s=1)


def normalize(x):
    return (x - x.min(0)) / x.ptp(0)


def correct_labels(x):
    for ex in x:
        if ex[-1] == 'False' or ex[-1] == 0:
            ex[-1] = -1
        else:
            ex[-1] = 1


def split_by_label(T):
    T1 = np.array([ex[1:-1] for ex in T if ex[-1] == -1])
    T2 = np.array([ex[1:-1] for ex in T if ex[-1] == 1])
    return T1, T2


def dataset_load(name):
    exset = get_data(f"/data/{name}/{name}")
    map_nominal_attr(exset)
    data = np.array(exset.examples)
    correct_labels(data)
    data = data.astype(np.float)
    T = normalize(data)
    correct_labels(T)
    return T
