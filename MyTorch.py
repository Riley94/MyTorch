import MyTorchCPP
import numpy as np
# need to rethink. not scalable
from MyTorchCPP import ones_like, zeros_like, rand_like, Dtype, rand, ones, zeros, from_numpy

class Tensor(MyTorchCPP.Tensor):
    def __init__(self, data):
        # Check if the data is a valid type
        if not isinstance(data, (list, tuple, MyTorchCPP.Tensor, np.ndarray)):
            raise TypeError(f"Expected a list, tuple, or MyTorch.Tensor, got {type(data).__name__}")

        if isinstance(data, list):
            if not all(isinstance(x, (int, float, list, tuple)) for x in data):
                raise ValueError("List should contain only numeric values or nested lists/tuples.")

        # Call the superclass constructor if validation passes
        super().__init__(data)