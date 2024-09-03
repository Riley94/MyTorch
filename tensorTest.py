import MyTensor

# Create Tensor instances using the Python interface
tensor1 = MyTensor.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2x3 Matrix
tensor2 = MyTensor.Tensor([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])  # 3x2 Matrix

# Perform inner product (matrix multiplication)
result = tensor1.innerProduct(tensor2)

# Print the result
print("Result of matrix multiplication:")
result.print()