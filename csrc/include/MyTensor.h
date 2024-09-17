#pragma once

#include <iostream>
#include <sstream>
#include <cassert>
#include <initializer_list>
#include <cstdint>
#include <vector>
#include "pybind_includes.h"

using namespace std;

enum class Dtype;

class Tensor {
private:
    vector<int64_t> shape; // Shape of the tensor
    vector<double> data;   // Flattened data array
    Dtype dtype;

    // Computes the total number of elements in the tensor based on its shape
    int64_t getFlatIndex(const vector<int64_t>& indices) const;

public:
    // Constructor to initialize the tensor with a given shape and fill it with zeros
    Tensor(const vector<int64_t>& shape);

    // Constructor to initialize the tensor with a given shape and data
    Tensor(const vector<int64_t>& shape, const initializer_list<double>& values);

    // accepting vectors
    Tensor(const vector<int64_t>& shape, const vector<double>& data, Dtype dtype);

    // python objects
    Tensor(const py::object& obj);

    // Get the shape of the tensor
    vector<int64_t> getShape() const { return shape; }

    // Get the number of elements in the tensor
    int64_t size() const { return data.size(); }

    // Access elements in the tensor (const version)
    double operator()(const vector<int64_t>& indices) const;

    // Access elements in the tensor (non-const version)
    double& operator()(const vector<int64_t>& indices);

    // Tensor addition (element-wise)
    Tensor operator+(const Tensor& other) const;

    // Tensor multiplication (element-wise)
    Tensor operator*(const Tensor& other) const;

    Tensor operator*(const double& other) const;

    // Tensor subtraction (element-wise)
    Tensor operator-(const Tensor& other) const;

    // Tensor division (element-wise)
    Tensor operator/(const Tensor& other) const;

    // Unary negation
    Tensor operator-() const;

    double& operator[](int64_t index);
    const double& operator[](int64_t index) const;

    Tensor slice(const py::object& row_obj, const py::object& col_obj) const;
    Tensor getitem(const py::object& obj) const;

    // Tensor dot product
    Tensor dot(const Tensor& other) const;

    Tensor transpose() const;
    py::array_t<double> numpy() const;

    //Tensor outer_product(const Tensor& other) const;

    // Print tensor elements
    string repr() const;

    void add_(const double& other);

    vector<double> get_data() const { return data; }
    vector<int64_t> get_shape() const { return shape; }
    Dtype get_dtype() const { return dtype; }

private:
    void printRecursive(ostream& os, const vector<int64_t>& indices, size_t dim) const;
    void parseList(const py::list& list, size_t depth = 0);
    
};