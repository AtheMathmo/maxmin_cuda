import torch
from maxmin import maxmin_extension

class MaxMinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, axis):
        outputs = maxmin_extension.forward(input, axis)
        return outputs

    @staticmethod
    def backward(ctx, grad_h):
        return grad_h, None

class MaxMin(torch.nn.Module):
    def __init__(self, axis=-1):
        super(MaxMin, self).__init__()
        self.axis = axis

    def forward(self, x):
       return MaxMinFunction.apply(x, self.axis)