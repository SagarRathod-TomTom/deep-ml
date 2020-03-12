import torch
from metrics.segmentation import Accuracy

if __name__ == "__main__":
    temp = torch.rand((2,3,3))
    temp = torch.tensor([[[0.7365, 0.8758, 0.9021],
                          [0.4410, 0.6723, 0.6516],
                          [0.0678, 0.3632, 0.1412]],

                         [[0.0976, 0.0659, 0.3631],
                          [0.1818, 0.4379, 0.2152],
                          [0.7521, 0.5383, 0.2609]]])
    target = torch.tensor([[[1,0,1],[0,0,1],[1,1,0]],[[1,0,1],[0,0,1],[1,1,0]]]).type(torch.FloatTensor)
    print(target)
    acc = Accuracy()
    print(acc(temp, target))