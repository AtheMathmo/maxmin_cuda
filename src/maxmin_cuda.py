import torch
import maxmin

class MaxMinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        outputs = maxmin.forward(input)
        return outputs

    @staticmethod
    def backward(ctx, grad_h):
        return None

class MaxMin(torch.nn.Module):
    def __init__(self, axis=-1):
        super(MaxMin, self).__init__()
        self.axis = axis

    def forward(self, x):
       return MaxMinFunction.apply(x)
