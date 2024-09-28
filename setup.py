import platform
import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob

headers = glob('csrc/include/*.h')

if platform.system() == 'Windows':
    # Windows OpenCL paths for parallel computing
    opencl_include_dir = r'C:\OpenCL-SDK\include'
    opencl_library_dir = r'C:\OpenCL-SDK\lib'
    opencl_libraries = ['OpenCL']
    extra_compile_args = ['/DCL_TARGET_OPENCL_VERSION=300']
elif platform.system() == 'Linux':
    # Linux paths
    opencl_include_dir = '/usr/include'
    opencl_library_dir = '/usr/lib'
    opencl_libraries = ['OpenCL']
    extra_compile_args = ['-DCL_TARGET_OPENCL_VERSION=300']
else:
    raise RuntimeError('Unsupported platform: ' + platform.system())


ext_modules = [
    Pybind11Extension(
        'MyTorchCPP',
        sources=['csrc/MyTensor.cpp', 'csrc/MyTorch.cpp', 'csrc/utils.cpp', 'csrc/Proxy.cpp', 'csrc/helpers.cpp'],
        include_dirs=['csrc/include', opencl_include_dir],
        library_dirs=[opencl_library_dir],
        libraries=opencl_libraries,
        depends=headers,
        extra_compile_args=extra_compile_args,
        cxx_std=20,  # Simplifies setting the C++ standard
    ),
]

setup(
    name='MyTorchCPP',
    version='0.1',
    author='Riley Bruce',
    author_email='rgbmrb@umsystem.edu',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    description='A Pytorch-like library in C++',
    packages=find_packages(),
    classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: Linux / Windows',
    ],
    python_requires='>=3.11',
    include_package_data=True,
)

# python setup.py build_ext --inplace
# python setup.py clean --all