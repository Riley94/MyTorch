from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

ext_modules = [
    Extension(
        'MyTorchCPP',
        sources=['csrc/MyTensor.cpp', 'csrc/MyTorch.cpp', 'csrc/utils.cpp', 'csrc/Proxy.cpp'],
        include_dirs=[
            'csrc/include',
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=['/std:c++20'] if sys.platform == 'win32' else ['-std=c++20'],
    ),
]

setup(
    name='MyTorchCPP',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)

# python setup.py build_ext --inplace
# python setup.py clean --all