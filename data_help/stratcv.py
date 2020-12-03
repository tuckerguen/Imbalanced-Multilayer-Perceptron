from data_help.data_help import split_by_label
from mlp.mlp import MLP, print_accs, eval_mlp
import numpy as np
from sklearn.model_selection import StratifiedKFold


def kfold_cv(mlp, T, k=5, test=None):
    """
    Perform k-fold cross validation on a classifier given a dataset
    :param mlp: MLP classifier
    :param T: dataset
    :param k: number of folds
    :param test: optional parameter to use a different test set
    :return: list of accuracy metrics
    """
    k = 5
    # Split T into feature vecs and labels
    TX = np.delete(T, np.s_[-1], axis=1)
    # TX = np.delete(TX, np.s_[0], axis=1)
    TY = np.delete(T, np.s_[:-1], axis=1)

    # List for accuracies over all folds
    all_acc = []
    # Split into stratified folds
    skf = StratifiedKFold(n_splits=k)
    # Train and test on each fold
    for train_idx, test_idx, in skf.split(TX, TY):
        # Create train and test sets, and split into classes
        X_train, X_test = TX[train_idx], TX[test_idx]
        Y_train, Y_test = TY[train_idx], TY[test_idx]
        # Add labels back on before splitting
        T_train = np.hstack((X_train, Y_train))
        T_test = np.hstack((X_test, Y_test))
        # Split
        train1, train2 = split_by_label(T_train)
        test1, test2 = split_by_label(T_test)

        # create new mlp
        if mlp.lambda_ != 0.5:
            mlp.lambda_ = len(train2) / len(T_train)
        mlp_curr = MLP(mlp.n, mlp.h, mlp.lambda_, mlp.hidden.afcn, mlp.mu, mlp.beta, mlp.repeat)
        # Train classifier on training set
        mlp_curr.train(train1, train2)

        # Evaluate the classifier on test set
        if test is None:
            acc = eval_mlp(mlp_curr, T_test, test1, test2)
        else:
            test1, test2 = split_by_label(test)
            acc = eval_mlp(mlp_curr, test, test1, test2)
        print(acc)
        all_acc.append(acc)

    # Print overall accuracies
    # final_accs = np.mean(all_acc, axis=0)
    # print_accs(final_accs)
    return all_acc
