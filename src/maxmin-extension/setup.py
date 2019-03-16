from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='maxmin',
      ext_modules=[CppExtension('maxmin', ['maxmin.cpp'])],
      cmdclass={'build_ext': BuildExtension})

# setup(name='maxmin_cuda',
#       ext_modules=[CUDAExtension('maxmin_cuda', ['maxmin_cuda.cpp', 'maxmin_cuda_kernel.cu'])],
#       cmdclass={'build_ext': BuildExtension})
