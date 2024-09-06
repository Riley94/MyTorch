#include "MyTensor.h"
#include "helpers.h"
#include "pybind_includes.h"

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
void Tensor::printRecursive(ostream& os, const vector<int64_t>& indices, size_t dim) const {
    if (dim == shape.size() - 1) {
        // Base case: printing the innermost dimension
        os << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            os << data[getFlatIndex(new_indices)];
            if (i < shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        // Recursive case: printing outer dimensions
        os << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            printRecursive(os, new_indices, dim + 1);
            if (i < shape[dim] - 1) {
                os << ",\n";
            }
        }
        os << "]";
    }
}

// Prints the tensor in a structured format
string Tensor::repr() const {
    ostringstream oss;
    oss << "Tensor(shape=[" << shape;
    oss << "], dtype=float32)\n"; // Assuming float32 for simplicity

    // Print the tensor data
    vector<int64_t> indices(shape.size(), 0);
    printRecursive(oss, {}, 0);  // Use a recursive method to print multi-dimensional arrays
    return oss.str();
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