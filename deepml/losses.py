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


class WeightedBCEWithLogitsLoss(torch.nn.Module):

    def __init__(self, w_p = None, w_n = None):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.w_p = w_p
        self.w_n = w_n

    def forward(self, logits, labels, epsilon = 1e-7):

        ps = torch.sigmoid(logits.squeeze())
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        loss = loss_pos + loss_neg
        return loss


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
         label should be 1 for positive pair image
         label should be 0 for negative image
        """
        embeddings1, embeddings2 = embeddings
        distance = self.distance_func(embeddings) if self.distance_func else F.pairwise_distance(embeddings1,
                                                                                                 embeddings2)
        label = self.label_transform(label) if self.label_transform else label

        pos = label * torch.pow(distance, 2)
        neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


class AngularPenaltySMLoss(torch.nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
