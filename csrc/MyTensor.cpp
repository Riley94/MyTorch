#include <typeinfo>

#include "MyTensor.h"
#include "helpers.h"
#include "pybind_includes.h"

Tensor::Tensor(const std::vector<int64_t>& shape) : shape(shape) {
    int64_t totalSize = numElements(shape);
    data = std::vector<double>(totalSize, 0.0); // Initialize all elements to zero
}

Tensor::Tensor(const std::vector<int64_t>& shape, const std::initializer_list<double>& values) : shape(shape) {
    int64_t totalSize = numElements(shape);
    assert(static_cast<int64_t>(values.size()) == totalSize);
    data = values;
}

Tensor::Tensor(const std::vector<int64_t>& shape, Dtype dtype) : shape(shape), dtype(dtype) {
    int64_t totalSize = numElements(shape);
    switch (dtype) {
    case Dtype::Float32:
        data = std::vector<float>(totalSize, 0.0f);
        break;
    case Dtype::Float64:
        data = std::vector<double>(totalSize, 0.0);
        break;
    case Dtype::Int32:
        data = std::vector<int32_t>(totalSize, 0);
        break;
    case Dtype::Int64:
        data = std::vector<int64_t>(totalSize, 0);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype provided");
    }
}

void Tensor::parseList(const py::list& list, size_t depth) {
    // Handle empty list
    if (py::len(list) == 0) {
        if (depth == 0) {
            shape.clear();
        }
        return;
    }

    // Update shape based on the current depth
    if (depth == shape.size()) {
        shape.push_back(py::len(list));
    } else {
        if (shape[depth] != static_cast<int64_t>(py::len(list))) {
            throw std::runtime_error("Inconsistent shape in nested list");
        }
    }

    for (auto item : list) {
        if (py::isinstance<py::list>(item)) {
            // Recursive call for nested lists
            parseList(item.cast<py::list>(), depth + 1);
        } else {
            if (!has_determined_dtype) {
                // First time encountering data, determine the dtype
                has_determined_dtype = true;

                if (py::isinstance<py::float_>(item)) {
                    dtype = Dtype::Float64;
                    data = std::vector<double>();
                } else if (py::isinstance<py::int_>(item)) {
                    int64_t value = item.cast<int64_t>();
                    if (value > std::numeric_limits<int32_t>::max() ||
                        value < std::numeric_limits<int32_t>::min()) {
                        dtype = Dtype::Int64;
                        data = std::vector<int64_t>();
                    } else {
                        dtype = Dtype::Int32;
                        data = std::vector<int32_t>();
                    }
                } else {
                    throw std::runtime_error("Unsupported data type in list");
                }
            }

            // Store the value into the appropriate vector in 'data'
            if (dtype == Dtype::Float64) {
                if (py::isinstance<py::float_>(item) || py::isinstance<py::int_>(item)) {
                    double val = item.cast<double>();
                    std::get<std::vector<double>>(data).push_back(val);
                } else {
                    throw std::runtime_error("Inconsistent data types in list");
                }
            } else if (dtype == Dtype::Int64) {
                if (py::isinstance<py::int_>(item)) {
                    int64_t val = item.cast<int64_t>();
                    std::get<std::vector<int64_t>>(data).push_back(val);
                } else {
                    throw std::runtime_error("Expected integer value in list");
                }
            } else if (dtype == Dtype::Int32) {
                if (py::isinstance<py::int_>(item)) {
                    int64_t val = item.cast<int64_t>();
                    if (val > std::numeric_limits<int32_t>::max() ||
                        val < std::numeric_limits<int32_t>::min()) {
                        throw std::runtime_error("Integer value out of range for int32_t");
                    }
                    std::get<std::vector<int32_t>>(data).push_back(static_cast<int32_t>(val));
                } else {
                    throw std::runtime_error("Expected integer value in list");
                }
            }
        }
    }
}

// Python objects
Tensor::Tensor(const py::object& obj) {
    // Handle case where a list is passed
    if (py::isinstance<py::list>(obj)) {
        auto list = obj.cast<py::list>();
        if (list.empty()) {
            // Handle empty tensor initialization
            shape = {};
            data = {};
        } else {
            parseList(list);
        }
    }
    // Handle case where a NumPy array is passed
    else if (py::isinstance<py::array_t<double>>(obj)) {
        auto np_array = obj.cast<py::array_t<double>>();
        auto buffer = np_array.request();
        shape = std::vector<int64_t>(buffer.shape.begin(), buffer.shape.end());
        data = std::vector<double>(static_cast<double*>(buffer.ptr), static_cast<double*>(buffer.ptr) + buffer.size);
    }
    else {
        throw std::invalid_argument("Tensor initialization error: Unsupported type or parameter combination.");
    }
}

int64_t Tensor::getFlatIndex(const std::vector<int64_t>& indices) const {
    assert(indices.size() == shape.size());
    int64_t flatIndex = 0;
    for (unsigned int i = 0; i < indices.size(); ++i) {
        assert(indices[i] >= 0 && indices[i] < shape[i]);
        flatIndex = flatIndex * shape[i] + indices[i];
    }
    return flatIndex;
}

template <typename T>
T Tensor::getValue(const std::vector<int64_t>& indices) const {
    int64_t flatIndex = getFlatIndex(indices);

    // Ensure that T matches the data type stored
    return std::visit([flatIndex](const auto& dataVec) -> T {
        using ValueType = typename std::decay_t<decltype(dataVec)>::value_type;

        if constexpr (!std::is_same_v<T, ValueType>) {
            throw std::runtime_error("Requested type does not match tensor's data type");
        }

        if (flatIndex < 0 || flatIndex >= static_cast<int64_t>(dataVec.size())) {
            throw std::out_of_range("Index out of range");
        }

        return dataVec[flatIndex];
    }, data);
}

Proxy Tensor::operator[](int64_t index) {
    return std::visit([index](auto& dataVec) -> Proxy {
        if (index < 0 || index >= static_cast<int64_t>(dataVec.size())) {
            throw std::out_of_range("Index out of range");
        }
        return Proxy(dataVec[index]);
    }, data);
}

template<typename T, typename Op>
void apply_elementwise_operation(Tensor& result, const auto& lhs_data, const auto& rhs_data, Op op) {
    size_t dataSize = lhs_data.size();
    std::vector<T> result_data(dataSize);

    for (size_t i = 0; i < dataSize; ++i) {
        result_data[i] = op(static_cast<T>(lhs_data[i]), static_cast<T>(rhs_data[i]));
    }

    result.set_data(result_data);
}

template<typename Op>
Tensor elementwise_binary_op(const Tensor& lhs, const Tensor& rhs, Op op, const std::string& op_name) {
    if (lhs.get_shape() != rhs.get_shape()) {
        throw std::runtime_error("Shape mismatch for " + op_name);
    }

    Dtype result_dtype = promote_types(lhs.get_dtype(), rhs.get_dtype());
    Tensor result(lhs.get_shape(), result_dtype);

    auto operationLambda = [&](const auto& lhs_data, const auto& rhs_data) {
        switch (result_dtype) {
            case Dtype::Float64:
                apply_elementwise_operation<double>(result, lhs_data, rhs_data, op);
                break;
            case Dtype::Float32:
                apply_elementwise_operation<float>(result, lhs_data, rhs_data, op);
                break;
            case Dtype::Int64:
                apply_elementwise_operation<int64_t>(result, lhs_data, rhs_data, op);
                break;
            case Dtype::Int32:
                apply_elementwise_operation<int32_t>(result, lhs_data, rhs_data, op);
                break;
            default:
                throw std::runtime_error("Unsupported data type for " + op_name);
        }
    };

    std::visit(operationLambda, lhs.get_data(), rhs.get_data());

    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return elementwise_binary_op(*this, other, std::plus<>(), "addition");
}

Tensor Tensor::operator-(const Tensor& other) const {
    return elementwise_binary_op(*this, other, std::minus<>(), "subtraction");
}

Tensor Tensor::operator*(const Tensor& other) const {
    return elementwise_binary_op(*this, other, std::multiplies<>(), "multiplication");
}

Tensor Tensor::operator/(const Tensor& other) const {
    return elementwise_binary_op(*this, other, std::divides<>(), "division");
}

/*Tensor Tensor::operator-() const {
    Tensor result(shape);
    for (int64_t i = 0; i < size(); ++i) {
        result.data[i] = -this->data[i];
    }
    return result;
}*/



/*Tensor Tensor::dot(const Tensor& other) const {
    checkDimensions(shape, other.shape);

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
}*/

void Tensor::printRecursive(std::ostream& os, const std::vector<int64_t>& indices, size_t dim) const {
    if (dim == shape.size()) {
        // Reached the full depth of indices, print the value
        int64_t flatIndex = getFlatIndex(indices);
        std::visit([&os, flatIndex](const auto& dataVec) {
            os << dataVec[flatIndex];
        }, data);
    } else {
        os << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            std::vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            printRecursive(os, new_indices, dim + 1);
            if (i < shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
        if (dim == 0) {
            os << "\n";
        }
    }
}

// Prints the tensor in a structured format
std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[" << shape;
    oss << "], dtype=";

    switch (dtype)
    {
    case Dtype::Float32:
        oss << "float32";
        break;
    case Dtype::Float64:
        oss << "float64";
        break;
    case Dtype::Int32:
        oss << "Int32";
        break;
    case Dtype::Int64:
        oss << "Int64";
        break;
    default:
        oss << "Unknown";
        break;
    }
    oss << ")\n";
    // Print the tensor data
    printRecursive(oss, {}, 0);  // Use a recursive method to print multi-dimensional arrays
    return oss.str();
}

// Implement the transpose method
/* Tensor Tensor::transpose() const {
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
} */

/* void Tensor::add_(const double& other)
{
    for (int64_t i = 0; i < size(); ++i) {
        data[i] += other;
    }
} */

/* py::array_t<double> Tensor::numpy() const {
    py::array_t<double> np_array(shape);
    auto buffer = np_array.request();
    double* ptr = static_cast<double*>(buffer.ptr);
    for (int64_t i = 0; i < size(); ++i) {
        ptr[i] = data[i];
    }
    return np_array;
} */

int64_t Tensor::size() const {
    return visit([](const auto& dataVec) -> int64_t {
        return static_cast<int64_t>(dataVec.size());
    }, data);
}

py::object Tensor::getItem(int64_t index) const {
    return std::visit([index](const auto& dataVec) -> py::object {
        using ValueType = typename std::decay_t<decltype(dataVec)>::value_type;

        if (index < 0 || index >= static_cast<int64_t>(dataVec.size())) {
            throw std::out_of_range("Index out of range");
        }

        ValueType value = dataVec[index];
        return py::cast(value);
    }, data);
}

void Tensor::setItem(int64_t index, py::object value) {
    std::visit([index, &value](auto& dataVec) {
        using ValueType = typename std::decay_t<decltype(dataVec)>::value_type;

        if (index < 0 || index >= static_cast<int64_t>(dataVec.size())) {
            throw std::out_of_range("Index out of range");
        }

        // Attempt to cast the Python object to the expected ValueType
        try {
            ValueType castedValue = value.cast<ValueType>();
            dataVec[index] = castedValue;
        } catch (const py::cast_error& e) {
            throw std::runtime_error("Type mismatch in assignment. " + std::string(e.what()));
        }
    }, data);
}