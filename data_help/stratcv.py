import random
from mldata import ExampleSet, Schema
from classifiers.confclassifier import ConfClassifier
import numpy as np
from scipy import stats


def make_folds(example_set):
    """
    Given an example set generate 5 random folds with even distributions of class labels
    :param example_set: Example set to create folds from
    :return: List of folds each of type ExampleSet
    """
    n_folds = 5
    folds = [ExampleSet(example_set.schema) for f in range(n_folds)]

    # To evenly distribute classes
    fold_ind_l0 = 0
    fold_ind_l1 = 0

    # Seed rng
    random.seed(12345)

    # Loop until no examples remaining
    while example_set:
        # Retrieve example randomly
        example_index = random.randrange(0, len(example_set))
        example = example_set.pop(example_index)
        label = example[-1]

        # For each label, add to fold and increment fold counter if not 
        # greater than num folds. Allows for even distribution of classes
        if label:
            folds[fold_ind_l1].examples.append(example)
            fold_ind_l1 = fold_ind_l1 + 1 if fold_ind_l1 < n_folds - 1 else 0
        else:
            folds[fold_ind_l0].examples.append(example)
            fold_ind_l0 = fold_ind_l0 + 1 if fold_ind_l0 < n_folds - 1 else 0

    return folds


def flatten(lol):
    """
    Convert a list of lists to a list
    :param lol: list of lists
    :return: Flattened list
    """
    return [val for sl in lol for val in sl]


def cross_validate(example_set, classifier):
    """
    Perform stratified cross validation on example set and
    given classifier object with train() and validate() methods
    :param example_set: Example set to create folds for training and validation on
    :param classifier: Classifier object to train and validate on folds
    :return: None
    """
    accuracies = []
    precisions = []
    recalls = []

    # Generate stratified cv folds
    folds = make_folds(example_set)
    confidence_labels = []
    for i in range(len(folds)):
        # Get training set for this iteration
        training_set = folds[i]
        # Get all features except index and class label
        valid_features = training_set.schema.features[1:-1]
        feature_schema = Schema(valid_features)

        # Train classifier on training set
        classifier.train(training_set)

        # Create test set from remaining folds
        test_set = ExampleSet(feature_schema)

        # Retrieve remaining folds
        test_set.examples = flatten(folds[1:]) if i == 0 else flatten(folds[:i - 1] + folds[i:])

        # Validate classifier on test set
        stats_conf = classifier.validate(test_set)

        if issubclass(type(classifier), ConfClassifier):
            confidence_labels.extend(stats_conf[1])
            accuracies.append(stats_conf[0][0])
            precisions.append(stats_conf[0][1])
            recalls.append(stats_conf[0][2])
        else:
            accuracies.append(stats_conf[0])
            precisions.append(stats_conf[1])
            recalls.append(stats_conf[2])

    precisions = [p for p in precisions if p]

    # sort
    if issubclass(type(classifier), ConfClassifier):
        confidence_labels = sorted(confidence_labels)
        true_labels = [pair[1] for pair in confidence_labels]
        tot_true = sum(true_labels)
        tot_false = len(confidence_labels) - tot_true
        unique_confidences = sorted(list(set([pair[0] for pair in confidence_labels])))
        tp_rate = 1
        fp_rate = 1
        AUC = 0
        for i, confidence in enumerate(unique_confidences):
            pred = [1 if pair[0] >= confidence else 0 for pair in confidence_labels]
            a = sum(pred)
            tp = sum([1 for i, label in enumerate(pred) if label == true_labels[i] and label])
            fp = sum([1 for i, label in enumerate(pred) if label != true_labels[i] and label])
            new_tp_rate = tp / tot_true
            new_fp_rate = fp / tot_false
            if new_fp_rate < fp_rate and new_tp_rate < tp_rate:
                AUC += abs(new_fp_rate - fp_rate) * (new_tp_rate + tp_rate) / 2
                tp_rate = new_tp_rate
                fp_rate = new_fp_rate
        print("Area under ROC = ", format(AUC, '.3f'))

    # Print results
    print("Accuracy = ", format(sum(accuracies) / len(folds), '.3f'), " ", format(np.std(accuracies), '.3f'))
    print("Precision = ", format(sum(precisions) / len(folds), '.3f'), " ", format(np.std(precisions), '.3f'))
    print("Recall = ", format(sum(recalls) / len(folds), '.3f'), " ", format(np.std(recalls), '.3f'))
    return accuracies


def ttest(base_acc, boosted_acc):
    error = [base_acc[i] - boosted_acc[i] for i in range(len(base_acc))]
    delta = np.average(error)
    s = np.std(error)
    # t = delta / (s * np.sqrt(2/N))
    # return 1 - stats.t.cdf(t, df=2*N - 2)
    t = 2.776 # t value for 95% CI w/ 5 folds (5-1=4 samples)
    if delta - t * s <= 0 <= delta + t * s:
        print("Accept Null Hypothesis")
    else:
        print("Reject Null Hypothesis")