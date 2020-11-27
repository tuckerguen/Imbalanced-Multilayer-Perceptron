from data_help.data_help import *
from MLP.mlp import MLP
import time
from data_help.exset_ops import get_data


def run():
    # num_samples = 1000
    # # ratio:1 positive to negative class ratio
    # ratio = 10
    # T1, T2, T = gen_data(num_samples, ratio)

    # np.save("T10.npy", T)
    # np.save("T110.npy", T1)
    # np.save("T210.npy", T2)

    # T1 = np.load("T110.npy")
    # T2 = np.load("T210.npy")
    # T = np.load("T10.npy")

    data = np.array(get_data("/Data/vehicles/vehicle").examples)
    correct_labels(data)
    data = data.astype(np.float)
    T = normalize(data)
    correct_labels(T)
    s = data.shape
    T1 = [ex[:-1] for ex in T if ex[-1] == -1]
    T2 = [ex[:-1] for ex in T if ex[-1] == 1]
    print(len(T1), len(T2), len(T))
    _lambda = len(T2) / len(T)
    print(_lambda)

    mlp = MLP(s[1]-1, 13, 0.5, np.tanh)
    # start_time = time.time()
    mlp.train(T1, T2)
    # np.save("hiddenweights", mlp.hidden.weights)
    # np.save("outputweights", mlp.output.weights)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # mlp.output.weights = np.load("outputweights.npy")
    # mlp.hidden.weights = np.load("hiddenweights.npy")

    p1 = [mlp.predict(ex) for ex in T1]
    print(p1)
    print("Num wrong: ", len([p for p in p1 if p < 0]), "/", len(T1))
    p2 = [mlp.predict(ex) for ex in T2]
    print(p2)
    print("Num wrong: ", len([p for p in p2 if p > 0]), "/", len(T2))

    # mlp.plot_decision_boundary()
    # plot_dataset(T)
    # plt.show()

if __name__ == '__main__':
    run()

# For lambda = 0.5, 10 hidden units
# Num wrong:  12 / 647
# Num wrong:  178 / 199

# Not doin that well on the vehicle set. Not enough training examples?
# Perhaps we can use the synthetic sampling technique not to increase
# minority class proportions but instead to increase number of training
# examples. Probably not this tho... Their performance is good on the same
# dataset

# 13 hidden nodes
# compute sqrt(Tpr*TNr)