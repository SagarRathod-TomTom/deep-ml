import torch


class JaccardLoss(torch.nn.modules.loss._Loss):

    def __init__(self, is_multiclass):
        super(JaccardLoss, self).__init__()
        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = torch.nn.Sigmoid()

    def forward(self, output, target):

        output = self.activation(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)

        jac = (intersection / (union - intersection + 1e-7)).mean()
        return 1 - jac