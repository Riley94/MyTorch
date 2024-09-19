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

// Type mapping from Dtype to C++ types
template <Dtype dtype>
struct DtypeToType;

template <>
struct DtypeToType<Dtype::Float64> {
    using type = double;
};

template <>
struct DtypeToType<Dtype::Float32> {
    using type = float;
};

template <>
struct DtypeToType<Dtype::Int64> {
    using type = int64_t;
};

template <>
struct DtypeToType<Dtype::Int32> {
    using type = int32_t;
};

int64_t numElements(const std::vector<int64_t>& shape);
Tensor ones_like(const Tensor& other, const Dtype& dtype = Dtype::Float64);
Tensor ones(const py::tuple& shape, const Dtype& dtype = Dtype::Float64);
Tensor zeros_like(const Tensor& other, const Dtype& dtype = Dtype::Float64);
Tensor zeros(const py::tuple& shape, const Dtype& dtype = Dtype::Float64);
Tensor rand_like(const Tensor& other, const Dtype& dtype = Dtype::Float64);
Tensor rand(const py::tuple& shape, const Dtype& dtype = Dtype::Float64);
Tensor from_numpy(const py::array& np_array);
Dtype promote_types(const Dtype& dtype1, const Dtype& dtype2);