import argparse
import math
import time

import unittest

import torch
from maxmin.maxmin_cuda import MaxMin as CudaMaxMin
from maxmin.maxmin_py import MaxMin as PyMaxMin


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
    
    def test_2d_0_axis(self):
        maxmin = CudaMaxMin(0)
        arr = torch.Tensor([[1, 2, 4, 5],[5, 7, 3, 2]])
        expected = torch.Tensor([[5, 7, 4, 5],[1, 2, 3, 2]])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())
    
    def test_2d_minus_axis(self):
        maxmin = CudaMaxMin(-1)
        arr = torch.Tensor([[1, 2, 4, 5],[1, 7, 3, 2]])
        expected = torch.Tensor([[2, 1, 5, 4],[7, 1, 3, 2]])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())
    
    def test_3d_minus_axis(self):
        maxmin = CudaMaxMin(-1)
        arr = torch.Tensor([[[1, 2], [4, 5],[1, 7], [3, 2]],
                            [[7, 3], [2, 5],[9, 7], [1, 2]]])
        expected = torch.Tensor([[[2, 1], [5, 4],[7, 1], [3, 2]],
                                 [[7, 3], [5, 2],[9, 7], [2, 1]]])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())
    
    def test_py_1d(self):
        maxmin = PyMaxMin(0)
        arr = torch.Tensor([1,2,4,5,7,3])
        expected = torch.Tensor([2,1,5,4,7,3])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())
    
    def test_py_2d(self):
        maxmin = PyMaxMin(1)
        arr = torch.Tensor([[1, 2, 4, 5],[1, 7, 3, 2]])
        expected = torch.Tensor([[2, 1, 5, 4],[7, 1, 3, 2]])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())
    
    def test_py_2d_0_axis(self):
        maxmin = PyMaxMin(0)
        arr = torch.Tensor([[1, 2, 4, 5],[5, 7, 3, 2]])
        expected = torch.Tensor([[5, 7, 4, 5],[1, 2, 3, 2]])
        arr = arr.cuda()
        expected = expected.cuda()
        actual = maxmin(arr)
        self.assertTrue((expected.cpu().numpy() == actual.cpu().numpy()).all())

    def test_py_vs_cuda(self):
        a = torch.randn((10, 8000)).cuda()
        py_maxmin = PyMaxMin(1)
        cuda_maxmin = CudaMaxMin(1)

        py_output = py_maxmin(a)
        cuda_output = cuda_maxmin(a)
        self.assertTrue((py_output.cpu().numpy() == cuda_output.cpu().numpy()).all())

    def test_py_vs_cuda_axis_0(self):
        a = torch.randn((4, 8000)).cuda()
        py_maxmin = PyMaxMin(0)
        cuda_maxmin = CudaMaxMin(0)

        py_output = py_maxmin(a)
        cuda_output = cuda_maxmin(a)
        self.assertTrue((py_output.cpu().numpy() == cuda_output.cpu().numpy()).all())
    
    def test_py_vs_cuda_3d_axis_1(self):
        a = torch.randn((40, 50, 800)).cuda()
        py_maxmin = PyMaxMin(1)
        cuda_maxmin = CudaMaxMin(1)

        py_output = py_maxmin(a)
        cuda_output = cuda_maxmin(a)
        self.assertTrue((py_output.cpu().numpy() == cuda_output.cpu().numpy()).all())

if __name__ == '__main__':
    unittest.main()

