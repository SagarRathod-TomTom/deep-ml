import torch
import torch.nn.functional as F
from .commons import true_positives, false_positives, false_negatives, multiclass_tp_fp_tn_fn


class Binarizer(torch.nn.Module):

    def __init__(self, threshold=0.5):
        super(Binarizer, self).__init__()
        self.threshold = threshold

    def forward(self, output):
        if output.ndim == 2 and output.shape[-1] > 1:
            # multiclass
            probabilities, indices = torch.max(F.softmax(output, dim=1), dim=1)
        else:
            # binary
            probabilities = torch.sigmoid(output)
            indices = torch.zeros_like(probabilities)
            indices[probabilities > self.threshold] = 1

        return indices, probabilities


class Accuracy(torch.nn.Module):

    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.binarize = Binarizer(threshold)

    def forward(self, output, target):
        indices, _ = self.binarize(output)
        return (indices == target).float().mean()


class Precision(torch.nn.Module):
    def __init__(self, threshold=0.5, epsilon=1e-6):
        super(Precision, self).__init__()
        self.binarize = Binarizer(threshold)
        self.epsilon = epsilon

    def forward(self, output, target):
        indices, probabilities = self.binarize(output)

        if output.shape[-1] > 1:
            # multiclass
            tp, fp, _, _ = multiclass_tp_fp_tn_fn(indices, target)
        else:
            tp = true_positives(indices, target)
            fp = false_positives(indices, target)

        return tp / (tp + fp + self.epsilon)


class Recall(torch.nn.Module):
    def __init__(self, threshold=0.5, epsilon=1e-6):
        super(Recall, self).__init__()
        self.binarize = Binarizer(threshold)
        self.epsilon = epsilon

    def forward(self, output, target):
        indices, probabilities = self.binarize(output)

        if output.shape[-1] > 1:
            # multiclass
            tp, _, _, fn = multiclass_tp_fp_tn_fn(indices, target)
        else:
            tp = true_positives(indices, target)
            fn = false_negatives(indices, target)

        return tp / (tp + fn + self.epsilon)


class FScore(torch.nn.Module):
    def __init__(self, beta=1.0, threshold=0.5, epsilon=1e-6):
        super(FScore, self).__init__()
        self.beta = beta
        self.binarize = Binarizer(threshold)
        self.epsilon = epsilon

    def forward(self, output, target):
        indices, probabilities = self.binarize(output)
        if output.shape[-1] > 1:
            # multiclass
            tp, fp, _, fn = multiclass_tp_fp_tn_fn(indices, target)
        else:
            tp = true_positives(indices, target)
            fp = false_positives(indices, target)
            fn = false_negatives(indices, target)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        return ((1 + self.beta ** 2) * precision * recall) / (self.beta ** 2 * (precision + recall))
