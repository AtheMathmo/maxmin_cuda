# CUDA implementation of the MaxMin activation function

This repository provides a CUDA implementation of the MaxMin activation function, via a pytorch extension.

The MaxMin activation was first introduced as the [OPLU unit](https://arxiv.org/abs/1604.02313), and later as a special case of the [GroupSort activation function](https://arxiv.org/abs/1811.05381). It is especially useful for neural networks which incorporate norm constraints (e.g. Lipschitz-constrained neural networks or orthogonal RNNs).

_The original pytorch GroupSort implementation can be found [here](https://github.com/cemanil/LNets), with a slightly newer version [here](https://github.com/ColinQiyangLi/LConvNet)._

## Installation

The extension can be installed simply with `python setup.py install`, and then subsequently imported as

```python
from maxmin import MaxMin
```

An implementation using standard pytorch ops is also available in `maxmin/maxmin_py`, and can be imported as

```python
from maxmin import PyMaxMin
```

## Benchmarking

Some quick benchmarking, applying MaxMin to a random matrix of size (50000,5000), and averaging over 5000 trials.

| Method        | CUDA?           | Fwd Avg Time (us)  | Bckwd Avg Time (us)  |
| ------------- |:-------------:| -----:|-----:|
| Default Pytorch | - [ ] | 365 | 286 |
| Default Pytorch | - [x] |   93  | 320 |
| This Repo       | - [ ] |  1004 | 1057 |
| This Repo       | - [x] |    37 | 247 |

In short, the implementation in this repo is moderately faster than the GPU implementation using default pytorch ops. However, the CPU implementation is slower than the default method (so if you need a good CPU implementation then use that one). It is also worth noting that the default implementation is still plenty fast for most practical purposes.

## Notes

- Improved error handling is needed on the C++ side.
- A specialized CUDA implementation of the more general GroupSort is harder to implement, and won't give substantial gains over the default pytorch operations. Therefore is isn't planned for now.


