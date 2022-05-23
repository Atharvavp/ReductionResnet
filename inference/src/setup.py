from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_average',
      ext_modules=[cpp_extension.CppExtension('custom_average', ['custom_average.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

def my_group_norm(g, input_64, input_128, input_256, input_512):
    return g.op("custom_namespace::custom_average", input_64, input_128, input_256, input_512)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('custom_namespace::custom_average', my_group_norm, 9)