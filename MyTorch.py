import MyTorchCPP
# need to rethink. not scalable
from MyTorchCPP import ones_like, zeros_like, rand_like, Dtype, rand, ones

class Tensor(MyTorchCPP.Tensor):
    def __init__(self, data):
        # Check if the data is a valid type
        if not isinstance(data, (list, tuple, MyTorchCPP.Tensor)):
            raise TypeError(f"Expected a list, tuple, or MyTorch.Tensor, got {type(data).__name__}")

        # Optionally, you can add more validation logic here
        if isinstance(data, list):
            if not all(isinstance(x, (int, float, list, tuple)) for x in data):
                raise ValueError("List should contain only numeric values or nested lists/tuples.")

        # Call the superclass constructor if validation passes
        super().__init__(data)