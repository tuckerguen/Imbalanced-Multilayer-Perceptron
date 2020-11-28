from data_help.data_help import *
from MLP.mlp import MLP, eval_mlp
from data_help.exset_ops import get_data
import sys


def dataset_load(name):
    data = np.array(get_data(f"/Data/{name}/{name}").examples)
    correct_labels(data)
    data = data.astype(np.float)
    T = normalize(data)
    correct_labels(T)
    T1 = np.array([ex[:-1] for ex in T if ex[-1] == -1])
    T2 = np.array([ex[:-1] for ex in T if ex[-1] == 1])
    return T, T1, T2


def make_file_str(name, _lambda):
    return f"{name}_{_lambda:.3f}"


def run():
    name = str(sys.argv[1])
    comp_lambda = bool(int(sys.argv[2]))
    old_weights = bool(int(sys.argv[3]))

    T, T1, T2 = dataset_load(name)
    shape = T1.shape

    if comp_lambda:
        _lambda = len(T2) / len(T)
    else:
        _lambda = 0.5

    file_base = make_file_str(name, _lambda)

    mlp = MLP(shape[1], 3, _lambda, np.tanh)

    if old_weights:
        mlp.hidden = np.load(f"weights/hiddenw_{file_base}")
        mlp.output = np.load(f"weights/outputw_{file_base}")
    else:
        mlp.train(T1, T2)

    eval_mlp(mlp, T, T1, T2, file_base)


if __name__ == '__main__':
    run()
