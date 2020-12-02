from SMOTE.smote import SMOTE_dataset, kmeans_SMOTE
from data_help.data_help import gen_data, split_by_label, plot_dataset
import numpy as np
import matplotlib.pyplot as plt


def main():
    T, T1, T2 = gen_data(1000, 2, 19)
    plot_dataset(T)
    plt.show()
    T_smotek, T2_smotek = kmeans_SMOTE(T, T2, 500, 5, int(len(T2)/15))
    T_smote, T2_smote = SMOTE_dataset(T, T2, 500, 5)
    plot_dataset(T_smotek)
    plt.show()
    plot_dataset(T_smote)
    plt.show()


if __name__ == '__main__':
    main()
