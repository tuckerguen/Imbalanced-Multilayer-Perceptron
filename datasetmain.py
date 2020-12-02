from data_help.data_help import split_by_label, dataset_load
from data_help.stratcv import kfold_cv
from mlp.mlp import MLP, eval_mlp, print_accs
import sys
import numpy as np
import matplotlib.pyplot as plt


def make_file_str(name, _lambda, h):
    return f"{name}_{_lambda:.3f}_{h}"


def run():
    name = str(sys.argv[1])
    cv = bool(int(sys.argv[2]))
    comp_lambda = bool(int(sys.argv[3]))
    old_weights = bool(int(sys.argv[4]))
    repeat = bool(int(sys.argv[5]))

    T = dataset_load(name)
    T1, T2 = split_by_label(T)

    mu = 0.1
    beta = 10
    h = 7

    # Compute lambda
    _lambda = len(T2) / len(T) / 2 if comp_lambda else 0.5
    # Create classifier
    mlp = MLP(T.shape[1]-2, h, _lambda, np.tanh, mu, beta, repeat)

    # Do stratified cross validation
    if cv:
        kfold_cv(mlp, T, 5)
    else:
        # Run once
        file_base = make_file_str(name, _lambda, h)

        # Load old weights
        if old_weights:
            mlp.hidden.weights = np.load(f"weights/hiddenw_{file_base}.npy")
            mlp.output.weights = np.load(f"weights/outputw_{file_base}.npy")
        else:
            # Train on dataset
            xs, loss_values = mlp.train(T1, T2)
            # Plot the loss
            loss_values = np.array(loss_values).reshape(len(xs), )
            plt.plot(xs, loss_values)
            plt.title(f"mu={mlp.mu}, beta={mlp.beta}")
            plt.show()
            # Save weights for this classifier
            np.save(f"weights/hiddenw_{file_base}", mlp.hidden.weights)
            np.save(f"weights/outputw_{file_base}", mlp.output.weights)

        # Print accuracies
        print("-" * 50)
        print(f"Accuracy for {h} hidden units")
        acc = eval_mlp(mlp, T, T1, T2)
        print_accs(acc)

        # Save accuracies
        np.savetxt(f"accuracies/accuracies_{file_base}", acc.reshape(len(acc), ))


if __name__ == '__main__':
    run()
