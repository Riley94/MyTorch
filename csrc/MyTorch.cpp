#include "MyTensor.h"
#include "utils.h"
#include "helpers.h"
#include "pybind_includes.h"

using namespace std;

#define BIND_OPERATOR(op_name, py_method, scalar_type) \
    .def(py_method, [](const Tensor& self, scalar_type scalar) { \
        return self op_name scalar; \
    }, py::is_operator())

// Pybind11 binding code
PYBIND11_MODULE(MyTorchCPP, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const vector<int64_t>&, const vector<double>&, Dtype>())
        .def(py::init<const py::object&>())
        // Define the property for `shape`
        .def_property_readonly("shape", &Tensor::get_shape)
        // Define the property for `dtype`
        .def_property_readonly("dtype", [](const Tensor &t) {
            return t.get_dtype();  // Assuming you have a method to return dtype
        })
        .def("dot", &Tensor::dot)
        .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator+))
        // Scalar addition
        // Bind __mul__ for int, float, double
        BIND_OPERATOR(*, "__add__", int)
        BIND_OPERATOR(*, "__add__", float)
        BIND_OPERATOR(*, "__add__", double)
        //.def("add_", &Tensor::add_) // In-place scalar addition

        // Element-wise multiplication
        .def("__mul__", [](const Tensor& self, const Tensor& other) {
            return self * other;  // Calls element-wise multiplication
        }, py::is_operator())
        // Scalar multiplication
        // Bind __mul__ for int, float, double
        BIND_OPERATOR(*, "__mul__", int)
        BIND_OPERATOR(*, "__mul__", float)
        BIND_OPERATOR(*, "__mul__", double)
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))
        .def("__neg__", static_cast<Tensor (Tensor::*)() const>(&Tensor::operator-))
        //.def("transpose", static_cast<Tensor (Tensor::*)() const>(&Tensor::transpose))
        //.def("numpy", &Tensor::numpy)
        .def("__repr__", &Tensor::repr)
        // Overload for single argument indexing (e.g., tensor[0])
        .def("__getitem__", &Tensor::getItem)
        .def("__setitem__", &Tensor::setItem);
        // Overload for two argument indexing (e.g., tensor[0, 1] or tensor[:, 0])
        //.def("__getitem__", &Tensor::slice)

    py::enum_<Dtype>(m, "Dtype")
        .value("Float32", Dtype::Float32)
        .value("Float64", Dtype::Float64)
        .value("Int32", Dtype::Int32)
        .value("Int64", Dtype::Int64)
        .export_values();

    m.def("rand_like", &rand_like, py::arg("other"), py::arg("dtype") = Dtype::Float64);
    m.def("rand", py::overload_cast<const py::tuple&, const Dtype&>(&rand), py::arg("shape"), py::arg("dtype") = Dtype::Float64);
    m.def("ones_like", &ones_like, py::arg("other"), py::arg("dtype") = Dtype::Float64);
    m.def("ones", &ones, py::arg("shape"), py::arg("dtype") = Dtype::Float64);
    m.def("zeros_like", &zeros_like, py::arg("other"), py::arg("dtype") = Dtype::Float64);
    m.def("zeros", &zeros, py::arg("shape"), py::arg("dtype") = Dtype::Float64);
    m.def("from_numpy", &from_numpy, py::arg("array"));
}

#undef BIND_OPERATOR