import unittest
import torch
from deepml.metrics.segmentation import Accuracy
from deepml.metrics import commons


class TestSegmentationMetrics(unittest.TestCase):

    def test_accuracy_binary(self):
        output = torch.tensor([[[0.7365, 0.8758, 0.9021],
                                [0.4410, 0.6723, 0.6516],
                                [0.0678, 0.3632, 0.1412]],

                               [[0.0976, 0.0659, 0.3631],
                                [0.1818, 0.4379, 0.2152],
                                [0.7521, 0.5383, 0.2609]]])

        target = torch.tensor([[[1, 0, 1], [0, 0, 1], [1, 1, 0]], [[1, 0, 1], [0, 0, 1], [1, 1, 0]]],
                              dtype=torch.float)
        accuracy = Accuracy()
        self.assertAlmostEqual(round(accuracy(output, target).item(), 4), 0.5556, delta=1e-4)


class TestImageClassificationMetrics(unittest.TestCase):

    def test_binary_classification(self):
        target = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.int8)
        output = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0], dtype=torch.int8)

        self.assertEqual(commons.true_positives(output, target), 4)
        self.assertEqual(commons.false_positives(output, target), 3)
        self.assertEqual(commons.false_negatives(output, target), 3)

    def test_multiclass_classification(self):
        target = torch.tensor([1, 4, 3, 2, 1, 1, 2, 3, 4, 2, 1, 2, 3, 4, 1, 2], dtype=torch.int8)
        output = torch.tensor([1, 2, 3, 4, 1, 2, 2, 3, 4, 2, 3, 1, 3, 4, 2, 3], dtype=torch.int8)

        self.assertEqual(commons.true_positives(output, target, is_multiclass=True), 9)
        self.assertEqual(commons.false_positives(output, target, is_multiclass=True), 7)
        self.assertEqual(commons.false_negatives(output, target, is_multiclass=True), 7)


if __name__ == "__main__":
    unittest.main()
