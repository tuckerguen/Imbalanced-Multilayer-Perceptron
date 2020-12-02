from SMOTE.smote import SMOTE_dataset
from data_help.data_help import gen_data, split_by_label, plot_dataset
import numpy as np
import matplotlib.pyplot as plt


def main():
    T, T1, T2 = gen_data(1000, 2, 19)
    plot_dataset(T)
    plt.show()
    T_smote, T2_smote = SMOTE_dataset(T, T2, 1000, 5, T1=T1, BLL=True)
    plot_dataset(T_smote)
    plt.show()


if __name__ == '__main__':
    main()
