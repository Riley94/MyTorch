import MyTensor
import numpy as np

class Tensor(MyTensor.Tensor):
    def __init__(self, data):
        super().__init__(data)


# Create Tensor instances using the Python interface
# --------------------------------------------------
# numpy array
test_array = np.array([[1, 2], [3, 4], [5, 6]])
tensor1 = Tensor(test_array)  # 3x2 Matrix
# List
tensor2 = Tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # 2x3 Matrix
print("Tensor1:")
tensor1.print()
print("Tensor2:")
tensor2.print()
# Perform inner product (matrix multiplication)
dot = tensor2.innerProduct(tensor1)

tensor3 = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3 Matrix
print("Tensor3:")
tensor3.print()

# addition
addit = tensor2 + tensor3

# subtraction
subt = tensor2 - tensor3

# multiplication
mult = tensor2 * tensor3

# Print the result
print("Result of matrix multiplication:")
dot.print()

print("Result of addition:")
addit.print()

print("Result of subtraction:")
subt.print()

print("Result of multiplication (element-wise):")
mult.print()