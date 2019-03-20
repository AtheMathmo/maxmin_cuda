import argparse
import math
import time

import torch
import maxmin_cuda

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cuda'])
parser.add_argument('-l', '--length', type=int, default=1000000)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from src.maxmin_py import MaxMin
elif options.example == 'cuda':
    from src.maxmin_cuda import MaxMin
    options.cuda = True

X = torch.randn((10, options.length // 10), requires_grad=True)

maxmin = MaxMin(0)

if options.cuda:
    X = X.cuda()

# Force CUDA initialization
forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0
for _ in range(options.runs):
    start = time.time()
    output = maxmin(X)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    loss = output.sum()
    start = time.time()
    loss.backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))