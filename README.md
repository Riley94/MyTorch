# MyTorch

MyTorch is intended as an educational project for myself wherein I attempt to reimpliment the PyTorch library functionalities for ML. I intend to implement everything from the Tensor object (partially complete) all the way to the NN functionality and optimizers. Most functionality will be written in C++ for efficiency, but just like PyTorch it will be exposed to Python using pybind11.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Building the Project](#building-the-project)
- [Contributing](#contributing)
- [License](#license)

## Features

- Initialize tensors from Python lists or NumPy arrays.
- Perform element-wise addition, subtraction, and multiplication of tensors.
- Compute dot products of tensors.
- Print tensor shapes and values.
- Lightweight and easy to extend.

## Dependencies

To build and use MyTensor, you need the following dependencies:

- **Python**: Latest Version
- **C++ Compiler**: A C++17 compliant compiler (e.g., `g++`).
- **Pybind11**: A lightweight header-only library for exposing C++ types in Python and vice versa.
- **NumPy**: A fundamental package for scientific computing with Python.
- **PyTorch**: What we are trying to emulate. Simply need this to show side-by-side comparison.
- **Graphviz**: For visualization in the model-building phase.

For testing you will need:

- **GoogleTest (gtest)**: latest version

## Installation
### The following assumes you're using Linux or Windows (with WSL)

**This installs and sets up necessary packages, including gtest and python and dev libs, and then creates a python venv in the current directory. (same as the following, except the pip installs)**

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Riley94/MyTorch.git
    cd MyTorch
    ```

2. **For easy setup on Linux, simply run:**
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    **If on Windows, start a Powershell window with admin priveleges and run:**
    ```bash
    .\setup.bat
    ```
    **Activate venv**:
    ```bash
    source ./bin/activate
    ```
    **On windows**
   ```bash
   .\Scripts\Activate.ps1
   ```
   or activate.bat if using cmd

4. **Install Dependencies**:

    Make sure you have Pybind and NumPy installed:

    ```bash
    pip install --upgrade numpy pybind11 torch graphviz build
    ```

    Install OpenCL SDK (new):

    On Linux:
    ```bash
    sudo apt install opencl-headers ocl-icd-opencl-dev -y
    ```

    On Windows:
    1. Get the latest release from https://github.com/KhronosGroup/OpenCL-SDK
    2. Extract contents to C:\OpenCL-SDK
    3. May need to add path to system environment variables
    

6. **Build the Project**:

    Use the provided Makefile to build the project:

    ```bash
    make
    ```

    This will generate the shared library file and whl for installing via pip.

7. **Use the Test Suite**

   ```bash
   make test
   ```

8. **Install via pip locally (for now)**

   ```bash
   pip install dist/*.whl --force-reinstall
   ```

   This is also in the first line of the example notebook (network.ipynb) for easy reinstall of new builds

9. **Clean Build**

   ```bash
   make clean
   ```

## Usage

See network.ipynb for proper usage.
