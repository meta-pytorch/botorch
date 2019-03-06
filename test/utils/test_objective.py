#! /usr/bin/env python3

import unittest

import torch
from botorch.utils import get_objective_weights_transform


class TestGetObjectiveWeightsTransform(unittest.TestCase):
    def testNoWeights(self):
        Y = torch.ones(5, 2, 4)
        objective_transform = get_objective_weights_transform(None)
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(Y, Y_transformed))

    def testOneWeightBroadcasting(self):
        Y = torch.ones(5, 2, 4)
        objective_transform = get_objective_weights_transform(torch.tensor([0.5]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(0.5 * Y.sum(dim=-1), Y_transformed))

    def testIncompatibleNumberOfWeights(self):
        Y = torch.ones(5, 2, 4)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 2.0]))
        with self.assertRaises(RuntimeError):
            objective_transform(Y)

    def testMultiTaskWeights(self):
        Y = torch.ones(5, 2, 4, 2)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(torch.sum(Y, dim=-1), Y_transformed))

    def testNoMCSamples(self):
        Y = torch.ones(2, 4, 2)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(torch.sum(Y, dim=-1), Y_transformed))
