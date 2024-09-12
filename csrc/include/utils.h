#pragma once

#include <random>
#include <vector>
#include "pybind_includes.h"

class Tensor; // Forward declaration

enum class Dtype {
    Float32,
    Float64,
    Int32,
    Int64
};

int64_t numElements(const std::vector<int64_t>& shape);
Tensor ones_like(const Tensor& other, Dtype dtype = Dtype::Float64);
Tensor ones_like(py::tuple& shape, Dtype dtype=Dtype::Float64);
Tensor zeros_like(const Tensor& other);
Tensor rand_like(const Tensor& other, Dtype dtype = Dtype::Float64);