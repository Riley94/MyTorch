#include "MyTensor.h"

using namespace std;

Tensor::Tensor(const vector<int64_t>& shape) : shape(shape) {
    int64_t totalSize = numElements(shape);
    data.resize(totalSize, 0.0); // Initialize all elements to zero
}

Tensor::Tensor(const vector<int64_t>& shape, const initializer_list<double>& values) : shape(shape) {
    int64_t totalSize = numElements(shape);
    assert(values.size() == totalSize);
    data = values;
}

Tensor::Tensor(const vector<int64_t>& shape, const std::vector<double>& data)
    : shape(shape), data(data) {
    assert(shape[0] * shape[1] == data.size()); // Ensuring data matches the shape
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
    for (long long unsigned int i = 0; i < indices.size(); ++i) {
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

// Print vector using << overloading (for shape). Source: https://academichelp.net/coding/cpp/how-to-print-a-vector-in-cpp.html#:~:text=To%20print%20a%20vector%20in%20C%2B%2B%2C%20you%20can%20use,each%20element%20in%20the%20vector.
template <typename ele_type>
std::ostream& operator<<(std::ostream& os, const std::vector<ele_type>& vect_name) {
    for (auto itr : vect_name) {
        if (itr == vect_name.back()) {
            os << itr;
        } else {
            os << itr << " ";
        }
    }
    return os; // Return the ostream object to allow chaining
}

// Prints the tensor in a structured format
void Tensor::print() const {
    cout << "Tensor with shape: [" << shape << "]"<< endl;
    printRecursive({}, 0);
    cout << endl;
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