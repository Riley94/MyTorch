#pragma once

#include <cstdint>
#include <vector>
#include <iostream>
#include "pybind_includes.h"

namespace mytorch {

// Helper template to trigger static_assert
template <typename>
inline constexpr bool always_false = false;

enum class Dtype {
    Float32,
    Float64,
    Int32,
    Int64
};

template <Dtype dtype>
struct DtypeToCppType;

template <>
struct DtypeToCppType<Dtype::Int32> {
    using type = int32_t;
};

template <>
struct DtypeToCppType<Dtype::Int64> {
    using type = int64_t;
};

template <>
struct DtypeToCppType<Dtype::Float32> {
    using type = float;
};

template <>
struct DtypeToCppType<Dtype::Float64> {
    using type = double;
};

// Helper alias
template <Dtype dtype>
using DtypeToCppType_t = typename DtypeToCppType<dtype>::type;

template<typename T1, typename T2>
struct PromoteType {
    using type = decltype(std::declval<T1>() + std::declval<T2>());
};

template<typename T1, typename T2>
using PromoteType_t = typename PromoteType<T1, T2>::type;

// promote the types of two tensors
Dtype promote_types(const Dtype& dtype1, const Dtype& dtype2);

// check if two shapes are compatible for dot product
void checkDimensions(const std::vector<int64_t>& shape, const std::vector<int64_t>& other_shape);

// return the total number of elements in a tensor
int64_t numElements(const std::vector<int64_t>& shape);

// get the dtype from a C++ type
template<typename T>
Dtype getDtypeFromCppType() {
    if constexpr (std::is_same_v<T, int32_t>) {
        return Dtype::Int32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return Dtype::Int64;
    } else if constexpr (std::is_same_v<T, float>) {
        return Dtype::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return Dtype::Float64;
    } else {
        throw std::invalid_argument("Unsupported data type provided");
    }
}

// promote the dtype of a tensor (scalar arithmetic)
template <typename ScalarType>
Dtype promote_dtype_with_scalar(const Dtype& tensor_dtype) {
    Dtype scalar_dtype;

    if constexpr (std::is_same_v<ScalarType, int32_t>) {
        scalar_dtype = Dtype::Int32;
    } else if constexpr (std::is_same_v<ScalarType, int64_t>) {
        scalar_dtype = Dtype::Int64;
    } else if constexpr (std::is_same_v<ScalarType, float>) {
        scalar_dtype = Dtype::Float32;
    } else if constexpr (std::is_same_v<ScalarType, double>) {
        scalar_dtype = Dtype::Float64;
    } else {
        static_assert(always_false<ScalarType>, "Unsupported scalar type");
    }

    return promote_types(tensor_dtype, scalar_dtype);
}

} // namespace mytorch