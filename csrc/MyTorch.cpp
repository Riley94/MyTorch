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
        .def(py::init([](const std::vector<int64_t>& shape, const std::string& dtype_str, const std::string& device_str = "cpu") {
            // Map dtype_str to Dtype enum
            Dtype dtype;
            if (dtype_str == "float32") {
                dtype = Dtype::Float32;
            } else if (dtype_str == "float64") {
                dtype = Dtype::Float64;
            } else if (dtype_str == "int32") {
                dtype = Dtype::Int32;
            } else if (dtype_str == "int64") {
                dtype = Dtype::Int64;
            } else {
                throw std::invalid_argument("Invalid dtype string: " + dtype_str);
            }

            // Map device_str to DeviceType enum
            DeviceType device;
            std::string device_str_lower = device_str;
            std::transform(device_str_lower.begin(), device_str_lower.end(), device_str_lower.begin(),
               [](unsigned char c){ return std::tolower(c); });
            if (device_str_lower == "cpu") {
                device = DeviceType::CPU;
            } else if (device_str_lower == "gpu") {
                device = DeviceType::GPU;
            } else {
                throw std::invalid_argument("Invalid device string: " + device_str);
            }

            return Tensor(shape, dtype, device);
        }), py::arg("shape"), py::arg("dtype"), py::arg("device") = "cpu")
        // Constructor with shape, data vector, dtype, and device
        .def(py::init([](const py::list& data_list, const std::string& dtype_str, const std::string& device_str = "cpu") {
            // Map dtype_str to Dtype enum
            Dtype dtype;
            if (dtype_str == "float32") {
                dtype = Dtype::Float32;
            } else if (dtype_str == "float64") {
                dtype = Dtype::Float64;
            } else if (dtype_str == "int32") {
                dtype = Dtype::Int32;
            } else if (dtype_str == "int64") {
                dtype = Dtype::Int64;
            } else {
                throw std::invalid_argument("Invalid dtype string: " + dtype_str);
            }

            // Map device_str to DeviceType enum
            DeviceType device;
            std::string device_str_lower = device_str;
            std::transform(device_str_lower.begin(), device_str_lower.end(), device_str_lower.begin(),
               [](unsigned char c){ return std::tolower(c); });
            if (device_str_lower == "cpu") {
                device = DeviceType::CPU;
            } else if (device_str_lower == "gpu") {
                device = DeviceType::GPU;
            } else {
                throw std::invalid_argument("Invalid device string: " + device_str);
            }

            return Tensor(data_list, dtype, device);
        }), py::arg("data"), py::arg("dtype"), py::arg("device") = "cpu")
        // Functions to be treated as properties
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("dtype", [](const Tensor &t) {
            return t.get_dtype();
        })
        .def_property_readonly("T", static_cast<Tensor (Tensor::*)() const>(&Tensor::transpose))
        .def_property_readonly("device", [](const Tensor &t) {
            return t.get_device_str();
        })
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
        .def("numpy", &Tensor::numpy)
        .def("__repr__", &Tensor::repr)
        // Overload for single argument indexing (e.g., tensor[0])
        .def("__getitem__", &Tensor::getItem)
        .def("__setitem__", &Tensor::setItem)
        // Overload for two argument indexing (e.g., tensor[0, 1] or tensor[:, 0])
        //.def("__getitem__", &Tensor::slice)
        .def("add_", &Tensor::add_<int>, py::arg("scalar")) // have to explicitly specify the types for template functions
        .def("add_", &Tensor::add_<float>, py::arg("scalar"))
        .def("add_", &Tensor::add_<double>, py::arg("scalar"));

    // Expose Dtype enum
    py::enum_<Dtype>(m, "Dtype")
        .value("Float32", Dtype::Float32)
        .value("Float64", Dtype::Float64)
        .value("Int32", Dtype::Int32)
        .value("Int64", Dtype::Int64)
        .export_values();

    // Expose DeviceType enum
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("GPU", DeviceType::GPU)
        .value("ALL", DeviceType::ALL)
        .export_values();

    m.def("rand_like", &rand_like, py::arg("other"), py::arg("dtype") = Dtype::Float64);
    m.def("rand", py::overload_cast<const py::tuple&, const Dtype&>(&rand), py::arg("shape"), py::arg("dtype") = Dtype::Float64); // use for overloaded functions
    m.def("ones_like", &ones_like, py::arg("other"), py::arg("dtype") = Dtype::Float64);
    m.def("ones", &ones, py::arg("shape"), py::arg("dtype") = Dtype::Float64);
    m.def("zeros_like", &zeros_like, py::arg("other"), py::arg("dtype") = Dtype::Float64);
    m.def("zeros", &zeros, py::arg("shape"), py::arg("dtype") = Dtype::Float64);
    m.def("from_numpy", &from_numpy, py::arg("np_array"));
    m.def("get_devices", &get_devices);
}

#undef BIND_OPERATOR