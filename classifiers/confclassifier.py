from classifiers.classifier import *
from exset_ops import *


class ConfClassifier(Classifier):
    def __init__(self):
        super().__init__()

    def confidence(self, ex: float) -> float:
        """
        Returns confidence that ex is the positive class
        """
        pass

    def validate(self, validation_set):
        """
        Compute accuracy, precision, recall of the trained classifier
        on the validation set
        """
        pred_labels = []
        confidence_labels = []
        all_labels = [ex[-1] for ex in validation_set]
        for i, ex in enumerate(validation_set):
            confidence = self.confidence(ex)
            label = self.classify(ex)
            pred_labels.append(label)
            confidence_labels.append([confidence, all_labels[i]])

        stats = [accuracy(pred_labels, all_labels), precision(pred_labels, all_labels), recall(pred_labels, all_labels)]
        return stats, confidence_labels
