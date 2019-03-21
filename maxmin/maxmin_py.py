import torch
import torch.nn as nn

class MaxMin(nn.Module):

    def __init__(self, axis=-1):
        super(MaxMin, self).__init__()
        self.axis = axis

    def forward(self, x):
        original_shape = x.size()
        num_units = x.size(self.axis) // 2
        size = process_maxmin_size(x, num_units, self.axis)
        sort_dim = self.axis if self.axis == -1 else self.axis + 1

        mins = torch.min(x.view(*size), sort_dim, keepdim=True)[0]
        maxes = torch.max(x.view(*size), sort_dim, keepdim=True)[0]

        maxmin = torch.cat((maxes, mins), dim=sort_dim)
        return maxmin.view(original_shape)


def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size


def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]
