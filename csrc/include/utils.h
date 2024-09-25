#pragma once

#include <random>
#include <vector>
#include "pybind_includes.h"

namespace mytorch {
    
class Tensor; // Forward declarations
enum class Dtype;

// return a tensor of ones with the same shape and dtype as the input tensor
Tensor ones_like(const Tensor& other, const Dtype& dtype);
// return a tensor of ones with the given shape and dtype
Tensor ones(const py::tuple& shape, const Dtype& dtype);
// return a tensor of zeros with the same shape and dtype as the input tensor
Tensor zeros_like(const Tensor& other, const Dtype& dtype);
// return a tensor of zeros with the given shape and dtype
Tensor zeros(const py::tuple& shape, const Dtype& dtype);
// return a tensor of random values with the same shape and dtype as the input tensor
Tensor rand_like(const Tensor& other, const Dtype& dtype);
// return a tensor of random values with the given shape and dtype
Tensor rand(const py::tuple& shape, const Dtype& dtype);
// create a tensor from a numpy array
template <typename T>
Tensor from_numpy(const py::array_t<T>& np_array);

} // namespace mytorch