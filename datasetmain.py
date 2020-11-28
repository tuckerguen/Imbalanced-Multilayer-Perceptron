from data_help.data_help import correct_labels, normalize, gen_Tk, plot_dataset
from mlp.mlp import MLP, eval_mlp, print_accs
from data_help.exset_ops import get_data, map_nominal_attr
import sys
import numpy as np
import matplotlib.pyplot as plt


def dataset_load(name):
    exset = get_data(f"/data/{name}/{name}")
    map_nominal_attr(exset)
    data = np.array(exset.examples)
    correct_labels(data)
    data = data.astype(np.float)
    T = normalize(data)
    correct_labels(T)
    return T


def make_file_str(name, _lambda, h):
    return f"{name}_{_lambda:.3f}_{h}"


def k_fold_cv(mlp, data, k=10):
    np.random.seed(123456)
    np.random.shuffle(data)
    folds = np.array_split(data, k)

    # Iterate through folds to determine test set
    for i in range(0, k):
        train_set = np.delete(folds, i, axis=0)
        train_set = np.concatenate(train_set)
        gen_Tk(train_set)
        mlp.train()
        test_set = folds[i]


def run():
    name = str(sys.argv[1])
    cv = bool(int(sys.argv[2]))
    comp_lambda = bool(int(sys.argv[3]))
    old_weights = bool(int(sys.argv[4]))

    T = dataset_load(name)

    mu = 0.1
    beta = 10
    h = 3

    # Do stratified cross validation
    if cv:
        k = 10
        np.random.seed(123456)
        np.random.shuffle(T)
        folds = np.array_split(T, k)

        all_acc = []

        # Iterate through folds to determine test set
        for i in range(0, k):
            # Create train set
            train_set = np.delete(folds, i, axis=0)
            train_set = np.concatenate(train_set)
            T1, T2 = gen_Tk(train_set)

            # Compute lambda
            if comp_lambda:
                _lambda = len(T2) / len(T)
            else:
                _lambda = 0.5

            # Train classifier on dataset
            mlp = MLP(T1.shape[1], h, _lambda, np.tanh, mu, beta)
            mlp.train(T1, T2)

            # Generate test set
            test_set = folds[i]
            test1, test2 = gen_Tk(test_set)

            # Evaluate the classifier
            acc = eval_mlp(mlp, test_set, test1, test2)
            if i == 0:
                all_acc = acc
            else:
                all_acc = np.vstack((all_acc, acc))

        # Print accuracies
        final_accs = np.mean(all_acc, axis=0)
        print_accs(final_accs)

    else:
        T1, T2 = gen_Tk(T)

        if comp_lambda:
            _lambda = len(T2) / len(T)
        else:
            _lambda = 0.5

        # for h in range(3, 27, 3):
        # for m in range(1, 10, 2):
        #     mu = 1 / m
        #     for beta in range(1, 20, 4):

        mlp = MLP(T1.shape[1], h, _lambda, np.tanh, mu, beta)
        file_base = make_file_str(name, _lambda, h)

        if old_weights:
            mlp.hidden.weights = np.load(f"weights/hiddenw_{file_base}.npy")
            mlp.output.weights = np.load(f"weights/outputw_{file_base}.npy")
        else:
            xs, loss_values = mlp.train(T1, T2)
            loss_values = np.array(loss_values).reshape(len(xs),)
            plt.plot(xs, loss_values)
            plt.title(f"mu={mlp.mu}, beta={mlp.beta}")
            plt.show()
            np.save(f"weights/hiddenw_{file_base}", mlp.hidden.weights)
            np.save(f"weights/outputw_{file_base}", mlp.output.weights)

        print("-"*50)
        print(f"Accuracy for {h} hidden units")
        acc = eval_mlp(mlp, T, T1, T2)
        print_accs(acc)
        np.savetxt(f"accuracies/accuracies_{file_base}", acc.reshape(len(acc), ))
        if T1.shape[1] == 2:
            mlp.plot_decision_boundary()
            plot_dataset(T)
            plt.title(f"{file_base} | {acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}, {acc[3]:.3f},"
                      f"{acc[4]:.3f}, {acc[5]:.3f}")
            plt.show()




if __name__ == '__main__':
    run()
