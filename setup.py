from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob

headers = glob('csrc/include/*.h')

ext_modules = [
    Pybind11Extension(
        'MyTorchCPP',
        sources=['csrc/MyTensor.cpp', 'csrc/MyTorch.cpp', 'csrc/utils.cpp', 'csrc/Proxy.cpp'],
        include_dirs=['csrc/include'],
        depends=headers,
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
    description='A short description of your package',
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