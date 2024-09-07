#include "MyTensor.h"
#include "utils.h"
#include "pybind_includes.h"

using namespace std;

// Forward declaration of util functions
Tensor ones_like(const Tensor& other);
Tensor zeros_like(const Tensor& other);
Tensor rand_like(const Tensor& other);

// Pybind11 binding code
PYBIND11_MODULE(MyTorch, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const vector<int64_t>&, const vector<double>&>())
        .def(py::init<const py::array_t<double>&>()) // Use specific type here
        .def(py::init<const py::list&>())
        .def("innerProduct", &Tensor::dot)
        .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator+))
        // Scalar multiplication
        .def("__mul__", [](const Tensor& self, double scalar) {
            return self * scalar;  // Calls scalar multiplication
        }, py::is_operator())

        // Element-wise multiplication
        .def("__mul__", [](const Tensor& self, const Tensor& other) {
            return self * other;  // Calls element-wise multiplication
        }, py::is_operator())
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))
        .def("__neg__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("transpose", static_cast<Tensor (Tensor::*)() const>(&Tensor::transpose))
        .def("__repr__", &Tensor::repr)
        // Overload for single argument indexing (e.g., tensor[0])
        .def("__getitem__", &Tensor::getitem)
        // Overload for two argument indexing (e.g., tensor[0, 1] or tensor[:, 0])
        .def("__getitem__", &Tensor::slice)
        .def("get_data", &Tensor::get_data)
        .def("get_shape", &Tensor::get_shape);

    m.def("ones_like", &ones_like);
    m.def("zeros_like", &zeros_like);
    m.def("rand_like", &rand_like);
}