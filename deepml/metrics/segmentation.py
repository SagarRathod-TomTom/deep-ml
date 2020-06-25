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


class Accuracy(torch.nn.Module):

    def __init__(self, is_multiclass=False, threshold=0.5):
        super(Accuracy, self).__init__()

        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = Binarizer(threshold, torch.nn.Sigmoid())

    def forward(self, output, target):

        output = self.activation(output)

        output = output.to(torch.float)
        target = target.to(torch.float)

        return (output == target).float().mean()


class IoU(torch.nn.Module):

    def __init__(self, is_multiclass=False, epsilon=1e-6):
        super(IoU, self).__init__()
        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = torch.nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, output, target):

        output = self.activation(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        
        # calculate mean over batch
        jac = (intersection / (union - intersection + self.epsilon)).mean()
        return jac

