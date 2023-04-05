import torch
import torch.nn.functional as F


class JaccardLoss(torch.nn.Module):
    """
    Jaccard Loss aka IoU

    """
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


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target) + self.eps)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0, distance_func=None, label_transform=None):
        """
        :param margin: the distance margin between positive and negative class
        :param distance_func: the distance function to use. By default is euclidean distance
        :param label_transform: transformation function to apply on target label if any
                                For example, using lambda function "lambda label: label[:, 0]"
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_func = distance_func
        self.label_transform = label_transform

    def forward(self, embeddings: torch.Tensor, label: torch.Tensor):
        """
         label should be zero for positive pair image
         label should be 1 for negative image
        """
        embeddings1, embeddings2 = embeddings
        distance = self.distance_func(embeddings) if self.distance_func else F.pairwise_distance(embeddings1,
                                                                                                 embeddings2)
        label = self.label_transform(label) if self.label_transform else label

        pos = (1 - label) * torch.pow(distance, 2)
        neg = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive
