#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cassert>
#include <initializer_list>
#include <cstdint>
#include <vector>


namespace py = pybind11;

using namespace std;

class Tensor {
private:
    vector<int64_t> shape; // Shape of the tensor
    vector<double> data;   // Flattened data array

    // Computes the total number of elements in the tensor based on its shape
    int64_t numElements(const vector<int64_t>& shape) const;

    int64_t getFlatIndex(const vector<int64_t>& indices) const;

public:
    // Constructor to initialize the tensor with a given shape and fill it with zeros
    Tensor(const vector<int64_t>& shape);

    // Constructor to initialize the tensor with a given shape and data
    Tensor(const vector<int64_t>& shape, const initializer_list<double>& values);

    // accepting vectors
    Tensor(const vector<int64_t>& shape, const std::vector<double>& data);

    // accepting python lists
    Tensor(const py::list& data);

    // accepting numpy arrays
    Tensor(const py::array_t<double>& data);

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

    // Tensor subtraction (element-wise)
    Tensor operator-(const Tensor& other) const;

    // Tensor division (element-wise)
    Tensor operator/(const Tensor& other) const;

    // Unary negation
    Tensor operator-() const;

    // Tensor dot product
    Tensor dot(const Tensor& other) const;

    Tensor transpose() const;

    //Tensor outer_product(const Tensor& other) const;

    // Print tensor elements
    void print() const;

private:
    void printRecursive(const vector<int64_t>& indices, size_t dim) const;
    void parseList(const py::list& list, size_t depth = 0);
    
};

// Pybind11 binding code
PYBIND11_MODULE(MyTensor, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, const std::vector<double>&>())
        .def(py::init<const py::array_t<double>&>()) // Use specific type here
        .def(py::init<const py::list&>())
        .def("innerProduct", &Tensor::dot)
        .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator+))
        .def("__mul__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator*))
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))
        .def("__neg__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("transpose", static_cast<Tensor (Tensor::*)() const>(&Tensor::transpose))
        .def("print", &Tensor::print);
}