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

- **Python**: Version 3.11
- **C++ Compiler**: A C++17 compliant compiler (e.g., `g++`).
- **Pybind11**: A lightweight header-only library for exposing C++ types in Python and vice versa.
- **NumPy**: A fundamental package for scientific computing with Python.

For testing you will need:

- **GoogleTest (gtest)**: latest version

## Installation
### The following assumes you're using Linux

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Riley94/MyTorch.git
    cd MyTorch
    ```

2. **Create Environment (Optional, but recommended)**:

    Make sure you have Python version 3.11 and pyvenv installed:

    ```bash
    python3.11 -m venv .
    ```

    Activate venv:

    ```bash
    source ./bin/activate
    ```

3. **Install Dependencies**:

    Make sure you have Pybind and NumPy installed:

    ```bash
    pip install numpy pybind11
    ```

4. **Build the Project**:

    Use the provided Makefile to build the project:

    ```bash
    make
    ```

    This will generate the shared library file `MyTorch.<extension>` that can be imported in Python.

5. **Use the Test Suite**

   ```bash
   make test
   ```

## Usage

Once built, you can import the `MyTensor` module in Python and start using it:

```python
import MyTorch

# Initialize a tensor from a Python list
tensor1 = MyTorch.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor2 = MyTorch.Tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

# Perform tensor addition
result = tensor1 + tensor2

# Print the result
print(result)

# Perform inner product
dot_product = tensor1.innerProduct(tensor2)
print("Dot product:", dot_product)
