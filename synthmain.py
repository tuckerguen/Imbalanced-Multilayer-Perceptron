from data_help.data_help import *
from mlp.mlp import MLP, eval_mlp
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


def train_and_plot(T, lambda_, title_base):
    T1, T2 = split_by_label_synth(T)
    # Create classifier
    mlp = MLP(T1.shape[1], 3, lambda_, np.tanh, 0.1, 10)
    mlp.train(T1, T2)
    acc = eval_mlp(mlp, T, T1, T2)
    print(acc)
    mlp.plot_decision_boundary()
    plot_dataset(T)
    plt.title(f"{title_base} | {acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}, {acc[3]:.3f},"
              f"{acc[4]:.3f}, {acc[5]:.3f}")
    plt.show()

def run():
    N = int(sys.argv[1])
    ratio = int(sys.argv[2])
    num_attr = int(sys.argv[3])
    load_data = bool(int(sys.argv[4]))
    comp_lambda = bool(int(sys.argv[5]))
    old_weights = bool(int(sys.argv[6]))
    repeat = bool(int(sys.argv[7]))

    # Load or generate dataset
    if load_data:
        T, T1, T2 = synth_load(N, ratio, num_attr)
    else:
        T, T1, T2 = synth_gen_and_save(N, ratio, num_attr)

    # Compute lambda
    _lambda = len(T2) / len(T) / 2 if comp_lambda else 0.5

    # Create file string base
    file_base = make_file_str(N, ratio, num_attr, _lambda)

    # Create classifier
    mlp = MLP(T1.shape[1], 3, _lambda, np.tanh, 0.1, 10, repeat)

    # Load old weights or train and save new weights
    if old_weights:
        mlp.hidden.weights = np.load(f"weights/hiddenw_{file_base}")
        mlp.output.weights = np.load(f"weights/outputw_{file_base}")
    else:
        mlp.train(T1, T2)
        np.save(f"weights/hiddenw_{file_base}", mlp.hidden.weights)
        np.save(f"weights/outputw_{file_base}", mlp.output.weights)

    # Evaluate classifier
    acc = eval_mlp(mlp, T, T1, T2)
    print(acc)
    # Plot decision boundary on dataset plot
    mlp.plot_decision_boundary()
    plot_dataset(T)
    plt.title(f"{file_base} | {acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}, {acc[3]:.3f},"
              f"{acc[4]:.3f}, {acc[5]:.3f}")
    plt.show()


def make_file_str(N, ratio, num_attr, _lambda):
    return f"{N}_{ratio}_{num_attr}_{_lambda:.3f}"


if __name__ == '__main__':
    run()
