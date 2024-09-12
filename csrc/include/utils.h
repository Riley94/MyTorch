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
Tensor ones_like(const Tensor& other, const Dtype& dtype = Dtype::Float64);
Tensor ones(const py::tuple& shape, const Dtype& dtype = Dtype::Float64);
Tensor zeros_like(const Tensor& other, const Dtype& dtype = Dtype::Float64);
Tensor zeros(const py::tuple& shape, const Dtype& dtype = Dtype::Float64);
Tensor rand_like(const Tensor& other, const Dtype& dtype = Dtype::Float64);
Tensor rand(const py::tuple& shape, const Dtype& dtype = Dtype::Float64);