import numpy as np
from time import time

numattrs = 2


def SMOTE(T, N, k):
    if N < 100:
        T = (N / 100) * T
        N = 100
    N = int(N / 100)

    np.random.seed(time())

    sample = [[1], [1]]
    newindex = 0
    synthetic = [[1], [1]]
    nnarray = []
    for i in range(T):
        # Compute k nearest neighbors for i
        K = knearestneighbors()
        nnarray.append(K)
        while N != 0:
            nn = np.random.randint(0, len(nnarray), size=1)
            for attr in range(numattrs):
                dif = sample[nnarray[nn]][attr] - sample[i][attr]
                gap = np.random.uniform(0, 1, size=1)
                synthetic[newindex][attr] = sample[i][attr] + gap * dif
            newindex += 1
            N = N - 1
