#pragma once

#include <iostream>
#include <sstream>
#include <cassert>
#include <initializer_list>
#include <cstdint>
#include <vector>
#include <variant>
#include "utils.h"
#include "Proxy.h"
#include "pybind_includes.h"

enum class Dtype;

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

    // Computes the total number of elements in the tensor based on its shape
    int64_t getFlatIndex(const std::vector<int64_t>& indices) const;

public:

    Proxy operator[](int64_t index);
    
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

    // Get the shape of the tensor
    std::vector<int64_t> getShape() const { return shape; }

    // Get the number of elements in the tensor
    int64_t size() const;

    template <typename T>
    T getValue(const std::vector<int64_t>& indices) const;

    // Tensor addition (element-wise)
    Tensor operator+(const Tensor& other) const;

    // Tensor multiplication (element-wise)
    Tensor operator*(const Tensor& other) const;

    template<typename T>
    Tensor operator*(const T& other) const;

    // Tensor subtraction (element-wise)
    Tensor operator-(const Tensor& other) const;

    // Tensor division (element-wise)
    Tensor operator/(const Tensor& other) const;

    // Unary negation
    //Tensor operator-() const;

    //Tensor slice(const py::object& row_obj, const py::object& col_obj) const;

    // Getter for item at index
    py::object getItem(int64_t index) const;

    // Setter for item at index
    void setItem(int64_t index, py::object value);

    // Tensor dot product
    //Tensor dot(const Tensor& other) const;

    //Tensor transpose() const;
    //py::array_t<double> numpy() const;

    //Tensor outer_product(const Tensor& other) const;

    // Print tensor elements
    std::string repr() const;

    //void add_(const double& other);

    //vector<double> get_data() const { return data; }
    std::vector<int64_t> get_shape() const { return shape; }
    Dtype get_dtype() const { return dtype; }
    tensorData get_data() const { return data; }

    void set_data(const tensorData& data) { this->data = std::move(data); }

private:
    void printRecursive(std::ostream& os, const std::vector<int64_t>& indices, size_t dim) const;
    void parseList(const py::list& list, size_t depth = 0);
    
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

template<typename T>
Tensor Tensor::operator*(const T& other) const {
    Tensor result(shape, dtype); // Retain the dtype of the current tensor

    std::visit([&](const auto& dataVec) {
        using ValueType = typename std::decay_t<decltype(dataVec)>::value_type;

        // Check if T can be converted to ValueType
        static_assert(std::is_arithmetic_v<T>, "Scalar must be an arithmetic type");
        if constexpr (!std::is_convertible_v<T, ValueType>) {
            throw std::runtime_error("Incompatible types for multiplication");
        }

        std::vector<ValueType> result_data(dataVec.size());
        for (size_t i = 0; i < dataVec.size(); ++i) {
            result_data[i] = dataVec[i] * static_cast<ValueType>(other);
        }
        result.data = std::move(result_data);
    }, data);

    return result;
}