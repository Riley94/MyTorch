#include <typeinfo>

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

double& Tensor::operator[](int64_t index) {
    assert(index >= 0 && index < static_cast<int64_t>(data.size()));
    return data[index];
}

const double& Tensor::operator[](int64_t index) const {
    assert(index >= 0 && index < static_cast<int64_t>(data.size()));
    return data[index];
}

// Overloaded method for single argument indexing
Tensor Tensor::getitem(const py::object& obj) const {
    cout << "getitem called" << endl;
    if (py::isinstance<py::int_>(obj)) {
        cout << "integer confirmed" << endl;
        // Single integer indexing (e.g., tensor[0])
        int64_t idx = obj.cast<int64_t>();
        if (shape.size() == 1) {
            cout << "Only one dimension confirmed" << endl;
            return Tensor({1}, {data[idx]});
        } else {
            cout << "More than one dimension confirmed" << endl;
            // Handle other dimensions (e.g., tensor[0] -> first row)
            std::vector<int64_t> result_shape = {shape[1]};
            std::vector<double> result_data(shape[1]);
            cout << "Starting loop" << endl;
            for (int64_t i = 0; i < shape[1]; ++i) {
                result_data[i] = data[idx * shape[1] + i];
            }
            cout << "Finished" << endl;
            return Tensor(result_shape, result_data);
        }
    } else if (py::isinstance<py::slice>(obj)) {
        // Handle slicing (e.g., tensor[:])
        py::ssize_t start, stop, step, length;
        py::slice slice = obj.cast<py::slice>();
        if (!slice.compute(shape[0], &start, &stop, &step, &length)) {
            throw std::runtime_error("Invalid slice");
        }
        // Implement slicing logic here for 1D slices
        std::vector<double> result_data(length);
        for (int64_t i = start; i < stop; i += step) {
            result_data[(i - start) / step] = data[i];
        }
        return Tensor({length}, result_data);
    }
    throw std::runtime_error("Invalid indexing type");
}

// Overloaded method for two arguments indexing (e.g., tensor[0, 1] or tensor[:, 0])
Tensor Tensor::slice(const py::object& row_obj, const py::object& col_obj) const {
    py::ssize_t row_start, row_stop, row_step, row_length;
    py::ssize_t col_start, col_stop, col_step, col_length;

    // Handle row slicing or single index
    if (py::isinstance<py::slice>(row_obj)) {
        py::slice row_slice = row_obj.cast<py::slice>();
        if (!row_slice.compute(shape[0], &row_start, &row_stop, &row_step, &row_length)) {
            throw std::runtime_error("Invalid row slice");
        }
    } else if (py::isinstance<py::int_>(row_obj)) {
        row_start = row_obj.cast<int64_t>();
        row_stop = row_start + 1;
        row_step = 1;
        row_length = 1;
    } else {
        throw std::runtime_error("Invalid row object: expected slice or int");
    }

    // Handle col slicing or single index
    if (py::isinstance<py::slice>(col_obj)) {
        py::slice col_slice = col_obj.cast<py::slice>();
        if (!col_slice.compute(shape[1], &col_start, &col_stop, &col_step, &col_length)) {
            throw std::runtime_error("Invalid col slice");
        }
    } else if (py::isinstance<py::int_>(col_obj)) {
        col_start = col_obj.cast<int64_t>();
        col_stop = col_start + 1;
        col_step = 1;
        col_length = 1;
    } else {
        throw std::runtime_error("Invalid col object: expected slice or int");
    }

    // Create a result tensor for the sliced portion
    vector<int64_t> result_shape = {row_length, col_length};
    Tensor result(result_shape);

    // Fill the result tensor with the appropriate data
    for (int64_t i = row_start; i < row_stop; i += row_step) {
        for (int64_t j = col_start; j < col_stop; j += col_step) {
            result({(i - row_start) / row_step, (j - col_start) / col_step}) = data[i * shape[1] + j];
        }
    }

    return result;
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

Tensor Tensor::operator*(const double& other) const {
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = this->data[i] * other;
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