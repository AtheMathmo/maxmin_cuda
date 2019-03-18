from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

# setup(name='maxmin',
#       ext_modules=[CppExtension('maxmin', ['maxmin.cpp'])],
#       cmdclass={'build_ext': BuildExtension})

setup(name='maxmin_cuda',
      version='0.0.1',
      description='CUDA kernels for MaxMin activtion function.',
      ext_modules=[CUDAExtension('maxmin_cuda', ['maxmin_cuda.cpp', 'maxmin_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})
