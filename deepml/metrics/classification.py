import torch
import torch.nn.functional as F
from .commons import true_positives, false_positives, false_negatives


class Accuracy(torch.nn.Module):

    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(self, output, target):

        if output.shape[-1] > 1:
            # multiclass
            _, indices = torch.max(F.softmax(output, dim=1), dim=1)
        else:
            # binary
            output = torch.sigmoid(output)
            indices = torch.zeros_like(output)
            indices[output > self.threshold] = 1

        indices = indices.to(torch.float)
        target = target.to(torch.float)

        return (indices == target).float().mean()


class Precision(torch.nn.Module):
    def __init__(self, threshold=0.5, epsilon=1e-6):
        super(Precision, self).__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, output, target):

        if output.shape[-1] > 1:
            # multiclass
            _, indices = torch.max(F.softmax(output, dim=1), dim=1)
            tp = true_positives(indices, target, is_multiclass=True)
            fp = false_positives(indices, target, is_multiclass=True)
        else:
            # binary
            output = torch.sigmoid(output)
            indices = torch.zeros_like(output)
            indices[output > self.threshold] = 1
            tp = true_positives(indices, target)
            fp = false_positives(indices, target)

        return tp / (tp + fp + self.epsilon)


class Recall(torch.nn.Module):
    def __init__(self, threshold=0.5, epsilon=1e-6):
        super(Recall, self).__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, output, target):

        if output.shape[-1] > 1:
            # multiclass
            _, indices = torch.max(F.softmax(output, dim=1), dim=1)
            tp = true_positives(indices, target, is_multiclass=True)
            fn = false_negatives(indices, target, is_multiclass=True)
        else:
            # binary
            output = torch.sigmoid(output)
            indices = torch.zeros_like(output)
            indices[output > self.threshold] = 1
            tp = true_positives(indices, target)
            fn = false_negatives(indices, target)

        return tp / (tp + fn + self.epsilon)
