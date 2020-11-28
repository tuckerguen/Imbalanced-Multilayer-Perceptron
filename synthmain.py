from data_help.data_help import *
from MLP.mlp import MLP, eval_mlp
import sys


def synth_load(N, ratio, num_attr):
    T = np.load(f"Data/synthetic/T_{N}_{ratio}_{num_attr}.npy")
    T1 = np.load(f"Data/synthetic/T1_{N}_{ratio}_{num_attr}.npy")
    T2 = np.load(f"Data/synthetic/T2_{N}_{ratio}_{num_attr}.npy")
    return T, T1, T2


def synth_gen_and_save(N, ratio, num_attr):
    T, T1, T2 = gen_data(N, num_attr, ratio)
    np.save(f"Data/synthetic/T_{N}_{ratio}_{num_attr}.npy", T)
    np.save(f"Data/synthetic/T1_{N}_{ratio}_{num_attr}.npy", T1)
    np.save(f"Data/synthetic/T2_{N}_{ratio}_{num_attr}.npy", T2)
    return T, T1, T2


def run():
    N = int(sys.argv[1])
    ratio = int(sys.argv[2])
    num_attr = int(sys.argv[3])
    load_data = bool(int(sys.argv[4]))
    comp_lambda = bool(int(sys.argv[5]))
    old_weights = bool(int(sys.argv[6]))

    if load_data:
        T, T1, T2 = synth_load(N, ratio, num_attr)
    else:
        T, T1, T2 = synth_gen_and_save(N, ratio, num_attr)

    if comp_lambda:
        _lambda = len(T2) / len(T)
    else:
        _lambda = 0.5

    shape = T1.shape
    file_base = make_file_str(N, ratio, num_attr, _lambda)

    mlp = MLP(shape[1], 3, _lambda, np.tanh)

    if old_weights:
        mlp.hidden.weights = np.load(f"weights/hiddenw_{file_base}")
        mlp.output.weights = np.load(f"weights/outputw_{file_base}")
    else:
        mlp.train(T1, T2)
        np.save(f"weights/hiddenw_{file_base}", mlp.hidden.weights)
        np.save(f"weights/outputw_{file_base}", mlp.output.weights)

    eval_mlp(mlp, T, T1, T2, file_base)


def make_file_str(N, ratio, num_attr, _lambda):
    return f"{N}_{ratio}_{num_attr}_{_lambda:.3f}"


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

# Synthetic dataset, ratio=10:1
# 3 hidden, lambda=0.5
# Majority class accuracy: 0.9306930693069307
# Minority class accuracy: 0.9111111111111111
# Overall accuracy 0.928928928928929

# lambda=0.09
# Majority class accuracy: 0.8921892189218922
# Minority class accuracy: 0.8666666666666667
# Overall accuracy 0.8898898898898899

# Ratio 19:1
# lambda = 1/19
# Majority class accuracy: 0.9284210526315789
# Minority class accuracy: 0.92
# Overall accuracy 0.928

# Ratio 19:1
# lambda = 0.5
# Majority class accuracy: 0.9263157894736842
# Minority class accuracy: 0.94
# Overall accuracy 0.927


# GREAT EXAMPLE
# Tk_19_618 dataset, 2 attr, 19:1 ratio
# w/ lambda = 1/19
# Majority class accuracy: 0.9126315789473685
# Minority class accuracy: 1.0
# Overall weighted accuracy 0.9956315789473684
# Overall accuracy 0.917

# w/ lambda = 0.5
# Majority class accuracy: 0.9210526315789473
# Minority class accuracy: 0.98
# Overall weighted accuracy 0.9505263157894737
# Overall accuracy 0.924
