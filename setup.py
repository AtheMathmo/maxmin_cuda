from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='maxmin',
      version='0.0.1',
      description='MaxMin activation function.',
      packages=['maxmin'],
      ext_modules=[CUDAExtension('maxmin.maxmin_extension', ['maxmin-extension/maxmin.cpp', 'maxmin-extension/maxmin_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})