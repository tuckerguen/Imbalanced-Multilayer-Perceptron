from SMOTE.smote import SMOTE_dataset, kmeans_SMOTE
from data_help.data_help import gen_data, split_by_label_synth, plot_dataset
from mlp.mlp import MLP, eval_mlp
import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt


def smote_validate_synth(T, T_smote, lambda_, title_base):
    T1_smote, T2_smote = split_by_label_synth(T_smote)
    T1, T2 = split_by_label_synth(T)
    # Create classifier
    mlp = MLP(T1.shape[1], 5, lambda_, np.tanh, 0.1, 10)
    # Train on random 80% of smote dataset
    np.random.shuffle(T1_smote)
    np.random.shuffle(T2_smote)
    mlp.train(T1_smote[:int(len(T1_smote)/1.25)], T2_smote[:int(len(T2_smote)/1.25)])
    # Evaluate on original dataset
    acc = eval_mlp(mlp, T, T1, T2)
    print(acc)
    mlp.plot_decision_boundary()
    plot_dataset(T)
    plt.title(f"{title_base} | {acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}, {acc[3]:.3f},"
              f"{acc[4]:.3f}, {acc[5]:.3f}")
    plt.show()


def main():
    T, T1, T2 = gen_data(1000, 2, 19)
    np.save("smote_0_1900_NEW", T)
    plot_dataset(T)
    plt.show()
    T_smotek, T2_smotek = kmeans_SMOTE(T, T2, 1900, 5, int(len(T2) / 15))
    np.save("smote_k_11pergroup_1900p_5nn_NEW", T_smotek)
    T_smote, T2_smote = SMOTE_dataset(T, T2, 1900, 5)
    np.save("smoke_normal_1900p_5nn_MEW", T_smote)
    plot_dataset(T_smotek)
    plt.show()
    plot_dataset(T_smote)
    plt.show()


if __name__ == '__main__':
    main()
