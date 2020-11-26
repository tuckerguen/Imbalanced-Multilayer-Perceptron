from data_help.data_help import *
from MLP.mlp import MLP
import time


def run():
    num_samples = 1000
    # ratio:1 positive to negative class ratio
    ratio = 19
    T1, T2, T = gen_data(num_samples, ratio)
    # plot_dataset(T)
    mlp = MLP(2, 3, 0.5, np.tanh)
    mlp.train(T1, T2)
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
    p1 = [mlp.predict(ex) for ex in T1]
    print(p1)
    print("Num wrong: ", len([p for p in p1 if p < 0]), "/", len(T1))
    p2 = [mlp.predict(ex) for ex in T2]
    print(p2)
    print("Num wrong: ", len([p for p in p2 if p > 0]), "/", len(T2))




if __name__ == '__main__':
    run()
