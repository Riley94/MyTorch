#pragma once

#include <iostream>
#include <sstream>
#include <cassert>
#include <initializer_list>
#include <cstdint>
#include <vector>
#include <variant>
#include "helpers.h"
#include "Proxy.h"
#include "pybind_includes.h"

namespace mytorch {

using tensorData = std::variant<std::vector<double>,
                    std::vector<float>, 
                    std::vector<int32_t>, 
                    std::vector<int64_t>>;

class Tensor {
private:
    std::vector<int64_t> shape; // Shape of the tensor
    tensorData data; // Data of the tensor
    Dtype dtype;

    bool has_determined_dtype = false; // Flag to check if dtype is already set in the parseList function

public:

    /* ------------------- Constructors ------------------- */
    
    // Constructor to initialize the tensor with a given shape and fill it with zeros
    Tensor(const std::vector<int64_t>& shape);

    // Constructor to initialize the tensor with a given shape and data
    Tensor(const std::vector<int64_t>& shape, const std::initializer_list<double>& values);

    // Constructor to initialize the tensor with a given shape and type
    Tensor(const std::vector<int64_t>& shape, Dtype dtype);

    // accepting vectors
    template <typename T>
    Tensor(const std::vector<int64_t>& shape, const std::vector<T>& dataVec, Dtype dtype);

    // python objects
    Tensor(const py::object& obj);

    /* --------------------- Operators --------------------- */

    // Tensor addition (element-wise)
    Tensor operator+(const Tensor& other) const;

    // Scalar addition
    template<typename ScalarType>
    Tensor operator+(const ScalarType& scalar) const;

    // Tensor multiplication (element-wise)
    Tensor operator*(const Tensor& other) const;

    template<typename ScalarType>
    Tensor operator*(const ScalarType& other) const;

    // Tensor subtraction (element-wise)
    Tensor operator-(const Tensor& other) const;

    // Scalar subtraction
    template<typename ScalarType>
    Tensor operator-(const ScalarType& scalar) const;

    // Tensor division (element-wise)
    Tensor operator/(const Tensor& other) const;

    // Scalar division
    template<typename ScalarType>
    Tensor operator/(const ScalarType& scalar) const;

    // Unary negation
    Tensor operator-() const;

    // Indexing operator
    Proxy operator[](int64_t index);

    /* ------------------------ Math ------------------------ */

    // Tensor dot product
    Tensor dot(const Tensor& other) const;

    template<typename ScalarType>
    void add_(const ScalarType& scalar);

    //Tensor outer_product(const Tensor& other) const;

    /* --------------------- Conversion --------------------- */

    //Tensor slice(const py::object& row_obj, const py::object& col_obj) const;

    Tensor transpose() const;

    py::array numpy() const;

    /* ---------------------- Display ---------------------- */

    // Print tensor elements
    std::string repr() const;

    

    /* ---------------------- Getters ---------------------- */

    // simple getters
    std::vector<int64_t> get_shape() const { return shape; }
    Dtype get_dtype() const { return dtype; }
    tensorData get_data() const { return data; }
    
    // Get the number of elements in the tensor
    int64_t size() const;

    // Getter for item at index
    py::object getItem(int64_t index) const;

    /* ---------------------- Setters ---------------------- */

    // simple setters
    void set_data(const tensorData& other_data) { this->data = std::move(other_data); }

    template <typename T>
    void set_data(const std::vector<T>& dataVec) { data = std::move(dataVec); }

    // Setter for item at index
    void setItem(int64_t index, py::object value);

private:
    // Helper function to print the tensor elements recursively
    void printRecursive(std::ostream& os, const std::vector<int64_t>& indices, size_t dim) const;
    // Helper function to parse a Python list and initialize the tensor
    void parseList(const py::list& list);
    // Computes the total number of elements in the tensor based on its shape
    int64_t getFlatIndex(const std::vector<int64_t>& indices) const;
    template <typename T>
    py::array_t<T> numpy_impl() const;
    
};

// had to define here due to linker error
template <typename T>
Tensor::Tensor(const std::vector<int64_t>& shape, const std::vector<T>& dataVec, Dtype dtype)
    : shape(shape), dtype(dtype) {
    // Ensure that T is one of the allowed types
    static_assert(
        std::is_same_v<T, double> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t>,
        "Unsupported data type");

    // Ensure that T matches the provided dtype
    if ((std::is_same_v<T, double> && dtype != Dtype::Float64) ||
        (std::is_same_v<T, float> && dtype != Dtype::Float32) ||
        (std::is_same_v<T, int32_t> && dtype != Dtype::Int32) ||
        (std::is_same_v<T, int64_t> && dtype != Dtype::Int64)) {
        throw std::invalid_argument("Data type does not match dtype");
    }

    // Ensure that the size of the data matches the expected size from the shape
    int64_t totalSize = numElements(shape);
    assert(static_cast<int64_t>(dataVec.size()) == totalSize);

    // Initialize the data variant with the appropriate vector type
    data = dataVec;
}

// helper function. had to define here
template<typename ScalarType, typename Operation>
Tensor scalar_operation(const Tensor& tensor, const ScalarType& scalar, Operation op) {
    // Determine the resulting dtype based on tensor's dtype and scalar's type
    Dtype result_dtype = promote_dtype_with_scalar<ScalarType>(tensor.get_dtype());

    // Create a new tensor with the promoted dtype
    Tensor result(tensor.get_shape(), result_dtype);

    std::visit([&](const auto& dataVec) {
        using ValueType = typename std::decay_t<decltype(dataVec)>::value_type;

        // Check if ScalarType can be converted to ValueType
        static_assert(std::is_arithmetic_v<ScalarType>, "Scalar must be an arithmetic type");
        if constexpr (!std::is_convertible_v<ScalarType, ValueType>) {
            throw std::runtime_error("Incompatible types for operation");
        }

        std::vector<ValueType> result_data(dataVec.size());
        for (size_t i = 0; i < dataVec.size(); ++i) {
            result_data[i] = op(dataVec[i], static_cast<ValueType>(scalar));
        }

        result.set_data(result_data);
    }, tensor.get_data());

    return result;
}

template<typename ScalarType>
Tensor Tensor::operator*(const ScalarType& scalar) const {
    return scalar_operation(*this, scalar, std::multiplies<>{});
}

template<typename ScalarType>
Tensor Tensor::operator+(const ScalarType& scalar) const {
    return scalar_operation(*this, scalar, std::plus<>{});
}

template<typename ScalarType>
Tensor Tensor::operator-(const ScalarType& scalar) const {
    return scalar_operation(*this, scalar, std::minus<>{});
}

template<typename ScalarType>
Tensor Tensor::operator/(const ScalarType& scalar) const {
    return scalar_operation(*this, scalar, std::divides<>{});
}

template<typename ScalarType>
void Tensor::add_(const ScalarType& scalar) {
    // Determine the resulting dtype based on tensor's dtype and scalar's type
    this->dtype = promote_dtype_with_scalar<ScalarType>(this->dtype);
    
    std::visit([&scalar](auto& dataVec) {
        using ValueType = typename std::decay_t<decltype(dataVec)>::value_type;

        // Check if ScalarType can be converted to ValueType
        static_assert(std::is_arithmetic_v<ScalarType>, "Scalar must be an arithmetic type");
        if constexpr (!std::is_convertible_v<ScalarType, ValueType>) {
            throw std::runtime_error("Incompatible types for operation");
        }

        for (auto& val : dataVec) {
            val += static_cast<ValueType>(scalar);
        }
    }, this->data);
}

} // namespace mytorch