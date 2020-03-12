import torch


class Binarizer:

    def __init__(self, threshold=0.5, activation=None, value=1):
        self.activation = activation
        self.threshold = threshold
        self.value = value

    def __call__(self, output: torch.FloatTensor):

        if self.activation is not None:
            output = self.activation(output)

        output[output >= self.threshold] = self.value
        output[output < self.threshold] = 0.0

        return output


class Accuracy(object):

    def __init__(self, is_multiclass=False, threshold=0.5):
        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = Binarizer(threshold, torch.nn.Sigmoid())

    def __call__(self, output, target):

        output = self.activation(output)

        output = output.to(torch.float)
        target = target.to(torch.float)

        return (output == target).float().mean()


class IoU(object):

    def __init__(self, is_multiclass=False):
        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = torch.nn.Sigmoid()

    def forward(self, output, target):

        output = self.activation(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        
        # calculate mean over batch
        jac = (intersection / (union - intersection + 1e-7)).mean()
        return jac

