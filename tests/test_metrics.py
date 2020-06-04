import unittest
import torch
from deepml.metrics.segmentation import Accuracy


class TestSegmentationMetrics(unittest.TestCase):

    def test_accuracy_binary(self):
        output = torch.tensor([[[0.7365, 0.8758, 0.9021],
                              [0.4410, 0.6723, 0.6516],
                              [0.0678, 0.3632, 0.1412]],

                             [[0.0976, 0.0659, 0.3631],
                              [0.1818, 0.4379, 0.2152],
                              [0.7521, 0.5383, 0.2609]]])

        target = torch.tensor([[[1,0,1],[0,0,1],[1,1,0]],[[1,0,1],[0,0,1],[1,1,0]]],
                              dtype=torch.float)
        accuracy = Accuracy()
        self.assertAlmostEqual(round(accuracy(output, target).item(), 4),  0.5556, delta=1e-4)


if __name__ == "__main__":
    unittest.main()