import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.random import Generator, PCG64
from sklearn.cluster import KMeans
import warnings


def kmeans_SMOTE(T, T2, p, k, n_clusters):
    # Partition T2 into clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(T2)
    # Create clusters from labels
    clusters = []
    for c in range(n_clusters):
        clusters.append([list(ex) for i, ex in enumerate(T2) if kmeans.labels_[i] == c])

    T2o = []
    # Apply smote to all clusters
    for cluster in clusters:
        addtl_samples = SMOTE(cluster, p, k)
        T2o.extend(addtl_samples)

    for ex in T2o:
        ex.append(-1)
    T2o = np.array(T2o)
    return np.vstack((T, T2o)), T2o


def SMOTE(T2, p, k, T1=None, BLL=False):
    """
    Synthetically oversample minority class examples
    :param T2: Dataset containing minority class examples
    :param p: % to oversample by
    :param k: number of nearest neighbors
    :param T1: majority class dataset
    :param BLL: use BLL SMOTE or not
    :return: (N/100)*T synthetic minority class samples
    """
    # Number of minority samples
    if T1 is None and BLL:
        warnings.warn("empty T1 with BLL=True will not work properly", UserWarning)

    N2 = len(T2)
    # Fit k-nearest neighbors for querying later
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(T2)

    # Account for N < 100
    # Later, it assumes that N%100 == 0
    if p < 100:
        np.random.shuffle(T2)
        N2 = int((p / 100) * N2)
        p = 100

    # Num synthetic samples to generate per minority sample
    synths_per_sample = int(p / 100)

    # Set of synthetic samples to return
    synthetic = []

    # Loop over all samples
    for i in range(N2):
        # Get ith sample
        t = T2[i]
        # array indices of k-nearest neighbors for t
        nnarray = knn.kneighbors([t])[1][0]
        # Array of nearest neighbor points
        nearest = [T2[i] for i in nnarray]
        # Generate synthetic samples for point t
        if BLL:
            new_samples = sample_synth_BLL(T1, t, nearest, synths_per_sample)
        else:
            new_samples = sample_synth(t, nearest, synths_per_sample)
        synthetic.extend(new_samples)
    return synthetic


def sample_synth(t, nearest, num_samples):
    """
    Generate num_samples samples for a given point and its set
    of nearest neighbors
    :param t: sample to generate points around
    :param nearest: array of k nearest neighbors
    :param num_samples: number of samples to generate
    :return set of synthetic samples
    """
    # Track count of samples generated
    num_generated = 0

    # number of attributes of each sample
    numattrs = len(nearest[0])
    # list of samples to return
    new_samples = []
    # Generate samples
    while num_samples != 0:
        new_sample = generate_sample(t, nearest, numattrs)
        new_samples.append(new_sample)
        # Track sample generation
        num_generated += 1
        num_samples -= 1

    return new_samples


def sample_synth_BLL(T1, t, nearest, num_samples):
    # Create nearest neigbor instance for T1
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(T1)

    # Track count of samples generated
    num_generated = 0
    # number of attributes of each sample
    numattrs = len(nearest[0])
    # list of samples to return
    new_samples = []

    # Generate samples
    while num_samples != 0:
        new_sample = generate_sample(t, nearest, numattrs)
        dj = [dist(new_sample, n) for n in nearest]
        ddiff = knn.kneighbors([new_sample])[0][0][0]
        print(ddiff)
        comp = [d <= ddiff for d in dj]
        print(comp)
        if all(comp):
            # accept sample
            new_samples.append(new_sample)
            # Track sample generation
            num_generated += 1
            num_samples -= 1

    return new_samples


def generate_sample(t, nearest, numattrs):
    # Seed randomizer
    rng = Generator(PCG64())
    # Choose a random nearest neighbor to t
    nn = rng.integers(len(nearest), size=1)[0]
    # new sample
    sample = []
    # Generate synthetic values for each attribute
    for attr in range(numattrs):
        # Difference between sample attr val and neighbor attr val
        dif = nearest[nn][attr] - t[attr]
        # Distance to move between sample and neighbor
        gap = rng.uniform(0, 1, size=1)[0]
        # Create synthetic sample (interpolated between sample and neighbor)
        sample.append(t[attr] + gap * dif)

    return sample


def dist(p1, p2):
    """
    Euclidean distance between two points
    :param p1: point 1
    :param p2: point 2
    :return: float euclidean distance
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** .5

def SMOTE_dataset(T, T2, p, k, T1=None, BLL=False):
    """
    Perform smote on minority class dataset and append to complete dataset
    :param T: complete dataset
    :param T2: minority class set
    :param p: percent SMOTE sample
    :param k: number nearest neighbors for smote
    :return: New, extended dataset, and synthetic sample set
    """
    T2o = SMOTE(T2, p, k, T1, BLL)
    # Add class labels
    for ex in T2o:
        ex.append(-1)
    T2o = np.array(T2o)
    return np.vstack((T, T2o)), T2o
