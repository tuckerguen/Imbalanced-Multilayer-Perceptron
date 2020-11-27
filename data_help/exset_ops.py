import math
from data_help.mldata import *
from bisect import bisect_left
import random
import datetime

"""
Collection of operations to perform on example sets/lists of examples
"""


def discretize_example_set(example_set, num_bins):
    """
    Given an example set, discretizes all continuous attributes into num_bins bins
    and reassigns example values into the index of the bin its value falls into
    :param example_set: ExampleSet of all examples
    :param num_bins: Number of bins to discretize each continuous attribute into
    :return: None
    """
    new_features = []

    for feat_idx, feat in enumerate(example_set.schema.features):
        if feat.type == Feature.Type.CONTINUOUS:
            # Collect all values of feature
            ex_vals = [ex[feat_idx] for ex in example_set.examples]
            # Create discrete bins for feature values (ordered)
            bins = discretize_cont_attr(ex_vals, num_bins)

            # Create new nominal feature with bin values
            feat_vals = list(range(0, len(bins) - 1))
            feat = Feature(feat.name, Feature.Type.NOMINAL, feat_vals)

            # Assign proper bin numbers for all examples
            for ex in example_set:
                val = ex[feat_idx]
                # Binary search for val's bin
                bin_num = bisect_left(bins, val) - 1
                ex[feat_idx] = bin_num

        # Add feature to list of new features
        new_features.append(feat)

    # Replace example schemas
    for ex in example_set:
        ex.schema.features = tuple(new_features)
    # Replace schema features
    example_set.schema.features = tuple(new_features)


def discretize_cont_attr(attr_ex_vals, num_bins):
    """
    Given list of example values for a continuous attribute,
    returns a list of bounds for num_bins bins
    :param attr_ex_vals: List of example values of attribute to discretize
    :param num_bins: Number of bins to discretize into
    :return: Sorted list of bounds of bins
    """
    attr_ex_vals.sort()
    num_in_bin = math.ceil(len(attr_ex_vals) / num_bins)
    one_extra = len(attr_ex_vals) % num_bins
    split_vals = [-math.inf]
    while len(attr_ex_vals) > num_in_bin:
        if one_extra:
            split_vals.append(attr_ex_vals[num_in_bin])
            del attr_ex_vals[:num_in_bin + 1]
            one_extra -= 1
        else:
            split_vals.append(attr_ex_vals[num_in_bin - 1])
            del attr_ex_vals[:num_in_bin]

    split_vals.append(math.inf)
    return split_vals


def map_nominal_attr(example_set):
    new_features = []

    # assign numbers for nominal values from 1 to k
    for feat_idx, feat in enumerate(example_set.schema.features):
        if feat.type == Feature.Type.NOMINAL and feat.type != Feature.Type.ID:
            new_vals = [feat.values.index(val) + 1 for val in feat.values]

            # Replace example schemas and values
            for ex in example_set:
                val_idx = feat.values.index(ex[feat_idx])
                ex[feat_idx] = new_vals[val_idx]

            # Set new feature
            feat = Feature(feat.name, Feature.Type.NOMINAL, new_vals)
        # Append updated feature to list of new features
        new_features.append(feat)

    # Replace example schemas
    for ex in example_set:
        ex.schema.features = tuple(new_features)

    # Replace schema features
    example_set.schema.features = new_features


def get_data(path):
    """
    Retrieve example set from file
    :param path: Relative path to data
    :return: ExampleSet from datafile at path
    """
    i = path.rfind('/')
    file_base = path[i + 1:]
    root = "." + path[:i + 1]
    return parse_c45(file_base, root)


def accuracy(pred, act):
    """
    Compute accuracy given sets of predicted class labels and true class labels
    :param pred: list of predicted class labels
    :param act: list of true class labels
    :return: accuracy = (TP + TN)/(TP+TN+FP+FN)
    """
    return sum([1 for i in range(0, len(act)) if pred[i] == act[i]]) / len(act)


def precision(pred, act):
    """
    Compute precision given sets of predicted class labels and true class labels
    :param pred: list of predicted class labels
    :param act: list of true class labels
    :return: precision = TP / (TP + FP)
    """
    if sum(pred) == 0:
        return None
    return sum([1 for i, label in enumerate(act) if pred[i] == label and label]) / sum(pred)


def recall(pred, act):
    """
    Compute recall given sets of predicted class labels and true class labels
    :param pred: list of predicted class labels
    :param act: list of true class labels
    :return: recall = TP / (TP + FN)
    """
    if sum(act) == 0:
        return None
    return sum([1 for i, label in enumerate(act) if pred[i] == label and label]) / sum(act)


def set_neg_labels(exset):
    """
    Set all negative class labels from 0 to -1
    """
    for ex in exset:
        ex[-1] = -1 if ex[-1] == 0 else 1


def add_noise(exset, p):
    """
    Invert class labels with a certain probability.
    Used to introduce levels of noise to dataset
    """
    random.seed(datetime.datetime.now())
    for ex in exset:
        r = random.uniform(0, 1)
        if r < p:
            ex[-1] = 1 if ex[-1] == 0 else 0
