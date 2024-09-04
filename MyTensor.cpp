#include "MyTensor.h"

using namespace std;

Tensor::Tensor(const vector<int64_t>& shape) : shape(shape) {
    int64_t totalSize = numElements(shape);
    data.resize(totalSize, 0.0); // Initialize all elements to zero
}

Tensor::Tensor(const vector<int64_t>& shape, const initializer_list<double>& values) : shape(shape) {
    int64_t totalSize = numElements(shape);
    assert(static_cast<int64_t>(values.size()) == totalSize);
    data = values;
}

Tensor::Tensor(const vector<int64_t>& shape, const vector<double>& data)
    : shape(shape), data(data) {
    assert(shape[0] * shape[1] == static_cast<int64_t>(data.size())); // Ensuring data matches the shape
}


void Tensor::parseList(const py::list& list, size_t depth) {
    if (depth == shape.size()) {
        shape.push_back(py::len(list));
    } else {
        if (shape[depth] != static_cast<int64_t>(py::len(list))) {
            throw std::runtime_error("Inconsistent shape in nested list");
        }
    }

    for (auto item : list) {
        if (py::isinstance<py::list>(item)) {
            parseList(item.cast<py::list>(), depth + 1);  // Recursive call for nested lists
        } else {
            data.push_back(item.cast<double>());
        }
    }
}

// Constructor for Python lists
Tensor::Tensor(const py::list& list) {
    cout << "Constructing tensor from Python list" << endl;
    parseList(list);
    cout << "Shape: ";
    for (auto s : shape) {
        cout << s << " ";
    }
    cout << endl;
}

// Constructor for NumPy arrays
Tensor::Tensor(const py::array_t<double>& np_array) {
    auto buffer = np_array.request();
    double* ptr = static_cast<double*>(buffer.ptr);
    shape = vector<int64_t>(buffer.shape.begin(), buffer.shape.end());
    data = vector<double>(ptr, ptr + buffer.size);
}

int64_t Tensor::numElements(const vector<int64_t>& shape) const {
    int64_t totalSize = 1;
    for (int64_t s : shape) {
        totalSize *= s;
    }
    return totalSize;
}

int64_t Tensor::getFlatIndex(const vector<int64_t>& indices) const {
    assert(indices.size() == shape.size());
    int64_t flatIndex = 0;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        assert(indices[i] >= 0 && indices[i] < shape[i]);
        flatIndex = flatIndex * shape[i] + indices[i];
    }
    return flatIndex;
}

double Tensor::operator()(const vector<int64_t>& indices) const {
    int64_t flatIndex = getFlatIndex(indices);
    return data[flatIndex];
}

double& Tensor::operator()(const vector<int64_t>& indices) {
    int64_t flatIndex = getFlatIndex(indices);
    return data[flatIndex];
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = this->data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = this->data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = this->data[i] / other.data[i];
    }
    return result;
}

Tensor Tensor::operator-() const {
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = -this->data[i];
    }
    return result;
}

void checkDimensions(const vector<int64_t>& shape, const vector<int64_t>& other_shape) {
    if (shape[1] != other_shape[0]) {
        stringstream ss;
        ss << "Dimensions mismatch for dot product. " << shape[1] << " != " << other_shape[0];
        throw runtime_error(ss.str());
    }
}

Tensor Tensor::dot(const Tensor& other) const {
    try {
        checkDimensions(shape, other.shape);
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
    }
    assert(shape[1] == other.shape[0]); // Columns of first must match rows of second

    vector<int64_t> result_shape = {shape[0], other.shape[1]};
    vector<double> result_data(result_shape[0] * result_shape[1], 0.0);

    for (int64_t i = 0; i < shape[0]; ++i) { // Loop over rows of first matrix
        for (int64_t j = 0; j < other.shape[1]; ++j) { // Loop over columns of second matrix
            for (int64_t k = 0; k < shape[1]; ++k) { // Loop over columns of first and rows of second
                result_data[i * result_shape[1] + j] += data[i * shape[1] + k] * other.data[k * other.shape[1] + j];
            }
        }
    }

    return Tensor(result_shape, result_data);
}

// Recursive function to print tensor data with appropriate formatting
void Tensor::printRecursive(const vector<int64_t>& indices, size_t dim) const {
    if (dim == shape.size() - 1) {
        // Base case: printing the innermost dimension
        cout << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            cout << data[getFlatIndex(new_indices)];
            if (i < shape[dim] - 1) {
                cout << ", ";
            }
        }
        cout << "]";
    } else {
        // Recursive case: printing outer dimensions
        cout << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            printRecursive(new_indices, dim + 1);
            if (i < shape[dim] - 1) {
                cout << ",\n";
            }
        }
        cout << "]";
    }
}

template <typename ele_type>
ostream& operator<<(ostream& os, const vector<ele_type>& vect_name) {
    os << "[";
    for (size_t i = 0; i < vect_name.size(); ++i) {
        os << vect_name[i];
        if (i != vect_name.size() - 1) {
            os << " "; // Space between elements
        }
    }
    os << "]";
    return os; // Return the ostream object to allow chaining
}

// Prints the tensor in a structured format
void Tensor::print() const {
    cout << "Tensor with shape: " << shape << endl;
    printRecursive({}, 0);
    cout << endl;
}

// Implement the transpose method
Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw runtime_error("Transpose is only supported for 2D tensors.");
    }

    vector<int64_t> transposed_shape = {shape[1], shape[0]};
    Tensor result(transposed_shape);

    for (int64_t i = 0; i < shape[0]; ++i) {
        for (int64_t j = 0; j < shape[1]; ++j) {
            result.data[j * shape[0] + i] = data[i * shape[1] + j];
        }
    }

    return result;
}

// For testing the Tensor class
int main() {
    // Create a 2x3 tensor initialized with zeros
    Tensor tensor1({2, 3});
    cout << "Tensor 1: (A)" << endl;
    tensor1.print(); // Should print: [[0 0 0],
                     //               [0 0 0]]

    cout << "Tensor 2: (B)" << endl;
    // Create a 2x3 tensor with specific values
    Tensor tensor2({2, 3}, {1, 2, 3, 4, 5, 6});
    tensor2.print(); // Should print: [[1 2 3],
                     //               [4 5 6]]

    // Element-wise addition
    cout << "Tensor 3: (A + B)" << endl;
    Tensor tensor3 = tensor1 + tensor2;
    tensor3.print(); // Should print: [[1 2 3],
                     //               [4 5 6]]

    // Element-wise multiplication
    cout << "Tensor 4: (B * B)" << endl;
    Tensor tensor4 = tensor2 * tensor2;
    tensor4.print(); // Should print: [[1 4 9],
                     //               [16 25 36]]

    // Element-wise subtraction
    cout << "Tensor 5: (B - A)" << endl;
    Tensor tensor5 = tensor2 - tensor1;
    tensor5.print(); // Should print: [[1 2 3],
                     //               [4 5 6]]

    // Element-wise division
    cout << "Tensor 6: (B / B)" << endl;
    Tensor tensor6 = tensor2 / tensor2;
    tensor6.print(); // Should print: [[1 1 1],
                     //               [1 1 1]]

    // Unary negation
    cout << "Tensor 7: (-B)" << endl;
    Tensor tensor7 = -tensor2;
    tensor7.print(); // Should print: [[-1 -2 -3],
                     //               [-4 -5 -6]]

    cout << "Tensor 8: (C)" << endl;
    // Create a 3x2 tensor with specific values
    Tensor tensor8({3, 2}, {1, 2, 3, 4, 5, 6});
    tensor8.print(); // Should print: [[1 2],
                     //               [3 4],
                     //               [5 6]]

    // Dot product
    cout << "Tensor 8: (B . C)" << endl;
    Tensor tensor9 = tensor2.dot(tensor8);
    tensor9.print(); // Should print: [[22 28],
                     //               [49 64]]

    // Accessing an element
    cout << tensor2({1, 2}) << endl; // Should print: 6
    tensor2({1, 2}) = 10;            // Setting the element at (1, 2) to 10
    tensor2.print();                 // Should print: 1 2 3 4 5 10

    return 0;
}