from exset_ops import *


class Classifier:
    """
    A general binary classifier (soft interface)
    """

    def __init__(self):
        self.weight = 1
        self.train_set = None

    def train(self, train_set):
        """
        Train the classifier, (soft) interface method
        """
        pass

    def classify(self, ex):
        """
        Classify an example, (soft) interface method
        """
        pass

    def validate(self, validation_set):
        """
        Run classifier on validation set and return performance metrics
        """
        # Predict class labels for all examples in example set
        pred_labels = self.classify_all(validation_set)
        all_labels = [ex[-1] for ex in validation_set]
        # print(pred_labels)
        stats = [accuracy(pred_labels, all_labels), precision(pred_labels, all_labels), recall(pred_labels, all_labels)]
        return stats

    def classify_all(self, ex_set):
        return list(map(self.classify, ex_set))

    @staticmethod
    def prepare_exset(exset):
        pass


def weighted_training_error(ex_set, classifications):
    """
    Weighted training error of this classifier
    """
    weights = []
    for i, ex in enumerate(ex_set):
        if ex[-1] != classifications[i]:
            weights.append(ex.weight)

    return sum(weights)