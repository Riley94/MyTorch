#include "MyTensor.h"
#include "utils.h"
#include "helpers.h"
#include "pybind_includes.h"

#define BIND_OPERATOR(op_name, py_method, scalar_type) \
    .def(py_method, [](const Tensor& self, scalar_type scalar) { \
        return self op_name scalar; \
    }, py::is_operator())

// Pybind11 binding code
PYBIND11_MODULE(MyTorchCPP, m) {
    using namespace mytorch;
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, const std::vector<double>&, Dtype>())
        .def(py::init<const py::object&>())
        // Functions to be treated as properties
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("dtype", [](const Tensor &t) {
            return t.get_dtype();
        })
        .def_property_readonly("T", static_cast<Tensor (Tensor::*)() const>(&Tensor::transpose))
        .def("dot", &Tensor::dot)
        // Element-wise addition
        .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator+))
        // Scalar addition
        BIND_OPERATOR(+, "__add__", int)
        BIND_OPERATOR(+, "__add__", float)
        BIND_OPERATOR(+, "__add__", double)
        // Element-wise multiplication
        .def("__mul__", [](const Tensor& self, const Tensor& other) {
            return self * other;  // Calls element-wise multiplication
        }, py::is_operator())
        // Scalar multiplication
        BIND_OPERATOR(*, "__mul__", int)
        BIND_OPERATOR(*, "__mul__", float)
        BIND_OPERATOR(*, "__mul__", double)
        // Element-wise subtraction
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        // Scalar subtraction
        BIND_OPERATOR(-, "__sub__", int)
        BIND_OPERATOR(-, "__sub__", float)
        BIND_OPERATOR(-, "__sub__", double)
        // Element-wise division
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))
        // Scalar division
        BIND_OPERATOR(/, "__truediv__", int)
        BIND_OPERATOR(/, "__truediv__", float)
        BIND_OPERATOR(/, "__truediv__", double)
        .def("__neg__", static_cast<Tensor (Tensor::*)() const>(&Tensor::operator-))
        //.def("numpy", &Tensor::numpy)
        .def("__repr__", &Tensor::repr)
        // Overload for single argument indexing (e.g., tensor[0])
        .def("__getitem__", &Tensor::getItem)
        .def("__setitem__", &Tensor::setItem);
        // Overload for two argument indexing (e.g., tensor[0, 1] or tensor[:, 0])
        //.def("__getitem__", &Tensor::slice)
        //.def("add_", &Tensor::add_) // In-place scalar addition

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