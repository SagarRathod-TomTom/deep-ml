import unittest
from deepml.train import Learner
from deepml.losses import RMSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision


class TestDataSet(torch.utils.data.Dataset):
    """ Class useful for reading images from a dataframe.
        Each row is assume to be the flattened array of an image.
        Each row is then reshaped to the provided image_size.
    """

    def __init__(self, samples=1000, img_size=(32, 32), channels=3,
                 num_classes=0):

        self.samples = samples
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes

    def __getitem__(self, index):

        x = torch.rand((self.channels, self.img_size[0], self.img_size[1]))

        if self.num_classes > 0:
            y = torch.randint(0, self.num_classes, (1,))
        else:
            y = torch.rand(1)

        return x, y

    def __len__(self):
        return self.samples


class TestNetwork(nn.Module):
    def __init__(self, head_nodes=10):
        super(TestNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, head_nodes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestLearner(unittest.TestCase):

    def test_learner_init(self):
        with self.assertRaises(ValueError):
            model = None
            optimizer = None
            model_save_path = 'test'
            learner = Learner(model, optimizer, model_save_path)

    @unittest.skip
    def test_image_regression(self):
        train_dataset = TestDataSet(samples=100)
        val_dataset = TestDataSet(samples=50)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8,
                                                 shuffle=False)

        model = TestNetwork(head_nodes=1)
        criterion = RMSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        learner = Learner(model, optimizer, 'test_img_reg', use_gpu=False)

        learner.fit(criterion, train_loader, val_loader, epochs=2)

    @unittest.skip
    def test_image_classification(self):
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True,
                                                     transform=torchvision.transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                   shuffle=True, num_workers=2)

        val_dataset = torchvision.datasets.CIFAR10(root='./data', download=True,
                                                   transform=torchvision.transforms.ToTensor())

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                                 shuffle=False, num_workers=2)

        model = TestNetwork(head_nodes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        learner = Learner(model, optimizer, 'test_img_class', use_gpu=False, classes=classes)

        learner.fit(criterion, train_loader, val_loader, epochs=2)


if __name__ == "__main__":
    unittest.main()
