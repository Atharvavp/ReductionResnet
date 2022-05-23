from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_average',
      ext_modules=[cpp_extension.CppExtension('custom_average', ['custom_average.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
