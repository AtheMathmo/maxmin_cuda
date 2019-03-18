import argparse
import math
import time

import unittest

import torch
from src.maxmin_cuda import MaxMin as CudaMaxMin


class TestMaxMin(unittest.TestCase):

    def test_1d(self):
        maxmin = CudaMaxMin(0)
        arr = torch.Tensor([1,2,4,5,7,3])
        expected = torch.Tensor([2,1,5,4,7,3])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())
    
    def test_2d(self):
        maxmin = CudaMaxMin(1)
        arr = torch.Tensor([[1, 2, 4, 5],[1, 7, 3, 2]])
        expected = torch.Tensor([[2, 1, 5, 4],[7, 1, 3, 2]])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())

if __name__ == '__main__':
    unittest.main()

