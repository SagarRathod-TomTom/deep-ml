import torch


class Binarizer(torch.nn.Module):

    def __init__(self, threshold=0.5, activation=None, value=1):
        super(Binarizer, self).__init__()
        self.activation = activation
        self.threshold = threshold
        self.value = value

    def forward(self, output: torch.FloatTensor):
        if self.activation is not None:
            output = self.activation(output)

        output[output >= self.threshold] = self.value
        output[output < self.threshold] = 0

        return output.to(torch.uint8)


def true_positives(output, target, is_multiclass=False):
    if is_multiclass:
        return (output == target).sum()
    else:
        return (output * target).sum()


def false_positives(output, target, is_multiclass=False):
    if is_multiclass:
        return (output != target).sum()
    else:
        return (output * (1 - target)).sum()


def false_negatives(output, target, is_multiclass=False):
    if is_multiclass:
        return (output != target).sum()
    else:
        return ((1 - output) * target).sum()



