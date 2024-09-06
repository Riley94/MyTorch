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
        .def("__mul__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator*))
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))
        .def("__neg__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("transpose", static_cast<Tensor (Tensor::*)() const>(&Tensor::transpose))
        .def("__repr__", &Tensor::repr)
        .def("get_data", &Tensor::get_data)
        .def("get_shape", &Tensor::get_shape);

    m.def("ones_like", &ones_like);
    m.def("zeros_like", &zeros_like);
    m.def("rand_like", &rand_like);
}

// For testing the Tensor class
/* int main() {
    // Create a 2x3 tensor initialized with zeros
    Tensor tensor1({2, 3});
    cout << "Tensor 1: (A)" << endl;
    tensor1.repr(); // Should print: [[0 0 0],
                     //               [0 0 0]]

    cout << "Tensor 2: (B)" << endl;
    // Create a 2x3 tensor with specific values
    Tensor tensor2({2, 3}, {1, 2, 3, 4, 5, 6});
    tensor2.repr(); // Should print: [[1 2 3],
                     //               [4 5 6]]

    // Element-wise addition
    cout << "Tensor 3: (A + B)" << endl;
    Tensor tensor3 = tensor1 + tensor2;
    tensor3.repr(); // Should print: [[1 2 3],
                     //               [4 5 6]]

    // Element-wise multiplication
    cout << "Tensor 4: (B * B)" << endl;
    Tensor tensor4 = tensor2 * tensor2;
    tensor4.repr(); // Should print: [[1 4 9],
                     //               [16 25 36]]

    // Element-wise subtraction
    cout << "Tensor 5: (B - A)" << endl;
    Tensor tensor5 = tensor2 - tensor1;
    tensor5.repr(); // Should print: [[1 2 3],
                     //               [4 5 6]]

    // Element-wise division
    cout << "Tensor 6: (B / B)" << endl;
    Tensor tensor6 = tensor2 / tensor2;
    tensor6.repr(); // Should print: [[1 1 1],
                     //               [1 1 1]]

    // Unary negation
    cout << "Tensor 7: (-B)" << endl;
    Tensor tensor7 = -tensor2;
    tensor7.repr(); // Should print: [[-1 -2 -3],
                     //               [-4 -5 -6]]

    cout << "Tensor 8: (C)" << endl;
    // Create a 3x2 tensor with specific values
    Tensor tensor8({3, 2}, {1, 2, 3, 4, 5, 6});
    tensor8.repr(); // Should print: [[1 2],
                     //               [3 4],
                     //               [5 6]]

    // Dot product
    cout << "Tensor 8: (B . C)" << endl;
    Tensor tensor9 = tensor2.dot(tensor8);
    tensor9.repr(); // Should print: [[22 28],
                     //               [49 64]]

    // Accessing an element
    cout << tensor2({1, 2}) << endl; // Should print: 6
    tensor2({1, 2}) = 10;            // Setting the element at (1, 2) to 10
    tensor2.repr();                 // Should print: 1 2 3 4 5 10

    return 0;
} */