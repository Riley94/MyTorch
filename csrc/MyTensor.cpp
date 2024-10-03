#include <typeinfo>
#include <iomanip>
#include <functional>

#include "MyTensor.h"
#include "OpenCLManager.h"
#include "pybind_includes.h"

namespace mytorch {

Tensor::Tensor(const std::vector<int64_t>& shape, DeviceType device) : shape(shape), device(device) {
    int64_t totalSize = numElements(shape);
    data = std::vector<double>(totalSize, 0.0); // Initialize all elements to zero
}

Tensor::Tensor(const std::vector<int64_t>& shape, const std::initializer_list<double>& values, DeviceType device) : shape(shape), device(device) {
    int64_t totalSize = numElements(shape);
    assert(static_cast<int64_t>(values.size()) == totalSize);
    data = values;
}

Tensor::Tensor(const std::vector<int64_t>& shape, Dtype dtype, DeviceType device) 
                                        : shape(shape), dtype(dtype), device(device) {
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

void Tensor::parseList(const py::list& list) {
    std::vector<int64_t> flat_data_int64;
    std::vector<double> flat_data_double;
    std::vector<float> flat_data_float;
    std::vector<int32_t> flat_data_int32;
    std::vector<int64_t> inferred_shape;

    std::function<void(const py::list&, size_t)> parse_helper;
    parse_helper = [&](const py::list& lst, size_t depth) {
        if (depth >= inferred_shape.size()) {
            inferred_shape.push_back(lst.size());
        } else {
            if (static_cast<long unsigned int>(inferred_shape[depth]) != lst.size()) {
                throw std::invalid_argument("Inconsistent dimensions in list for Tensor initialization.");
            }
        }

        for (auto item : lst) {
            if (py::isinstance<py::list>(item)) {
                parse_helper(item.cast<py::list>(), depth + 1);
            }
            else if (dtype == Dtype::Float32) {
                float val = item.cast<float>();
                flat_data_float.push_back(static_cast<float>(val));
            }
            else if (dtype == Dtype::Float64) {
                double val = item.cast<double>();
                flat_data_double.push_back(static_cast<double>(val));
            }
            else if (dtype == Dtype::Int32) {
                int32_t val = item.cast<int32_t>();
                flat_data_int32.push_back(static_cast<int32_t>(val));
            }
            else if (dtype == Dtype::Int64) {
                int64_t val = item.cast<int64_t>();
                flat_data_int64.push_back(static_cast<int64_t>(val));
            }
            else {
                throw std::invalid_argument("Unsupported data type in list for Tensor initialization.");
            }
        }
    };

    // Start parsing
    parse_helper(list, 0);

    // Set the shape
    shape = inferred_shape;
    
    switch (dtype)
    {
    case Dtype::Float32:
        set_data(flat_data_float);
        break;
    case Dtype::Float64:
        set_data(flat_data_double);
        break;
    case Dtype::Int32:
        set_data(flat_data_int32);
        break;
    case Dtype::Int64:
        set_data(flat_data_int64);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype provided");
        break;
    }
}

// Python objects
Tensor::Tensor(const py::object& obj, const Dtype& dtype, const DeviceType& device) 
    : dtype(dtype), device(device) {

    // Handle case where a list is passed
    if (py::isinstance<py::list>(obj)) {
        auto list = obj.cast<py::list>();
        if (list.empty()) {
            // Handle empty tensor initialization
            this->shape = {};
            this->data = {};
        } else {
            parseList(list);
        }
    }
    // Handle case where a NumPy array is passed
    else if (py::isinstance<py::array>(obj)) {
        auto np_array = obj.cast<py::array>().attr("copy")(py::arg("order") = 'C').cast<py::array>(); // ensure contiguous
        auto buffer = np_array.request();

        // Extract shape information
        shape = std::vector<int64_t>(buffer.shape.begin(), buffer.shape.end());

        switch (dtype)
        {
        case Dtype::Float32:
            {
                std::vector<float> tensor_data(reinterpret_cast<float*>(buffer.ptr),
                                                reinterpret_cast<float*>(buffer.ptr) + buffer.size);
                set_data(tensor_data);
                break;
            }
        case Dtype::Float64:
            {  
                std::vector<double> tensor_data(reinterpret_cast<double*>(buffer.ptr),
                                                reinterpret_cast<double*>(buffer.ptr) + buffer.size);
                set_data(tensor_data);
                break;
            }
        case Dtype::Int32:
            {
                std::vector<int32_t> tensor_data(reinterpret_cast<int32_t*>(buffer.ptr),
                                                reinterpret_cast<int32_t*>(buffer.ptr) + buffer.size);
                set_data(tensor_data);
                break;
            }
        case Dtype::Int64:
            {
                std::vector<int64_t> tensor_data(reinterpret_cast<int64_t*>(buffer.ptr),
                                                reinterpret_cast<int64_t*>(buffer.ptr) + buffer.size);
                set_data(tensor_data);
                break;
            }
        default:
            throw std::invalid_argument("Unsupported dtype provided");
        }
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

Proxy Tensor::operator[](int64_t index) {
    return std::visit([index](auto& dataVec) -> Proxy {
        if (index < 0 || index >= static_cast<int64_t>(dataVec.size())) {
            throw std::out_of_range("Index out of range");
        }
        return Proxy(dataVec[index]);
    }, data);
}

template<typename T, typename Op>
void apply_elementwise_operation(Tensor& result, const auto& lhs_data, const auto& rhs_data, Op op, bool is_division = false) {
    size_t dataSize = lhs_data.size();
    std::vector<T> result_data(dataSize);

    for (size_t i = 0; i < dataSize; ++i) {
        // Special handling for division to prevent integer division
        if (is_division) {
            result_data[i] = op(static_cast<double>(lhs_data[i]), static_cast<double>(rhs_data[i]));
        } else {
            result_data[i] = op(static_cast<T>(lhs_data[i]), static_cast<T>(rhs_data[i]));
        }
    }

    result.set_data(result_data);
}

std::string generateElementwiseKernel(const std::string& op_name, Dtype dtype) {
    std::string typeStr;
    switch (dtype) {
        case Dtype::Float64:
            typeStr = "double";
            break;
        case Dtype::Float32:
            typeStr = "float";
            break;
        case Dtype::Int64:
            typeStr = "long";
            break;
        case Dtype::Int32:
            typeStr = "int";
            break;
        default:
            throw std::runtime_error("Unsupported data type for OpenCL kernel generation");
    }

    std::string operation;
    if (op_name == "addition") {
        operation = "lhs[i] + rhs[i]";
    } else if (op_name == "subtraction") {
        operation = "lhs[i] - rhs[i]";
    } else if (op_name == "multiplication") {
        operation = "lhs[i] * rhs[i]";
    } else if (op_name == "division") {
        operation = "lhs[i] / rhs[i]";
    } else {
        throw std::runtime_error("Unsupported operation for OpenCL kernel generation");
    }

    std::string kernelCode = R"(
    __kernel void elementwise_op(__global const TYPE* lhs, __global const TYPE* rhs, __global TYPE* result) {
        int i = get_global_id(0);
        result[i] = OPERATION;
    }
    )";

    // Replace placeholders
    size_t pos;
    while ((pos = kernelCode.find("TYPE")) != std::string::npos) {
        kernelCode.replace(pos, 4, typeStr);
    }
    while ((pos = kernelCode.find("OPERATION")) != std::string::npos) {
        kernelCode.replace(pos, 9, operation);
    }

    return kernelCode;
}

void perform_opencl_elementwise_op(const Tensor& lhs, const Tensor& rhs, Tensor& result, const std::string& op_name) {
    // Ensure tensors are on the device
    lhs.ensureOnDevice();
    rhs.ensureOnDevice();
    result.ensureOnDevice();

    auto& context = OpenCLManager::getInstance().getContext(); // using GPU as default
    auto& queue = OpenCLManager::getInstance().getQueue(); // using GPU as default

    // Define the OpenCL kernel code
    std::string kernelCode = generateElementwiseKernel(op_name, result.get_dtype());

    // Build the OpenCL program
    cl::Program::Sources sources;
    sources.push_back({kernelCode.c_str(), kernelCode.length()});

    cl::Program program(context, sources);
    try {
        program.build();
    } catch (const cl::Error& e) {
        // Print build errors
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
        std::cerr << "Error building OpenCL program: " << e.what() << "(" << e.err() << ")\n";
        std::cerr << "Build log:\n" << buildLog << "\n";
        throw;
    }

    // Create the kernel
    cl::Kernel kernel(program, "elementwise_op");

    // Set kernel arguments
    kernel.setArg(0, lhs.get_clBuffer());
    kernel.setArg(1, rhs.get_clBuffer());
    kernel.setArg(2, result.get_clBuffer());

    // Execute the kernel
    size_t globalSize = lhs.size();  // Total number of elements
    cl::NDRange global(globalSize);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

    // Optionally wait for completion
    queue.finish();
}

template<typename Op>
Tensor elementwise_binary_op(const Tensor& lhs, const Tensor& rhs, Op op, const std::string& op_name) {
    if (lhs.get_shape() != rhs.get_shape()) {
        throw std::runtime_error("Shape mismatch for " + op_name);
    }

    Dtype result_dtype;
    bool is_division = (op_name == "division");


    // Always promote to Float32 for division, unless Float64 is needed
    if (is_division) {
        if (lhs.get_dtype() == Dtype::Float64 || rhs.get_dtype() == Dtype::Float64) {
            result_dtype = Dtype::Float64;
        } else {
            result_dtype = Dtype::Float32;
        }
    } else {
        result_dtype = promote_types(lhs.get_dtype(), rhs.get_dtype());
    }

    Tensor result(lhs.get_shape(), result_dtype);

    // Check if computation should be on GPU
    if (lhs.get_device() == DeviceType::GPU && rhs.get_device() == DeviceType::GPU) {
        // Perform computation using OpenCL
        result.set_device(DeviceType::GPU);
        perform_opencl_elementwise_op(lhs, rhs, result, op_name);
    } else {
        // Perform computation on CPU
        auto operationLambda = [&](const auto& lhs_data, const auto& rhs_data) {
            switch (result_dtype) {
                case Dtype::Float64:
                    apply_elementwise_operation<double>(result, lhs_data, rhs_data, op, is_division);
                    break;
                case Dtype::Float32:
                    apply_elementwise_operation<float>(result, lhs_data, rhs_data, op, is_division);
                    break;
                case Dtype::Int64:
                    apply_elementwise_operation<int64_t>(result, lhs_data, rhs_data, op, is_division);
                    break;
                case Dtype::Int32:
                    apply_elementwise_operation<int32_t>(result, lhs_data, rhs_data, op, is_division);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type for " + op_name);
            }
        };
        std::visit(operationLambda, lhs.get_data(), rhs.get_data());
    }
    
    result.readFromDevice();
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

Tensor Tensor::operator-() const {
    Tensor output(shape, dtype);
    std::visit([&](auto&& dataVec) {
        using VecType = std::decay_t<decltype(dataVec)>;
        VecType result_data(dataVec.size());
        for (size_t i = 0; i < dataVec.size(); ++i) {
            result_data[i] = -dataVec[i];
        }
        output.set_data(result_data);
    }, data);
    return output;
}

template <typename ResultType, typename LhsType, typename RhsType>
void compute_dot_product(
    const std::vector<int64_t>& shape, 
    const std::vector<int64_t>& other_shape, 
    const LhsType& lhs_data, 
    const RhsType& rhs_data, 
    std::vector<ResultType>& result_data) 
{
    for (int64_t i = 0; i < shape[0]; ++i) { // Loop over rows of the first matrix
        for (int64_t j = 0; j < other_shape[1]; ++j) { // Loop over columns of the second matrix
            for (int64_t k = 0; k < shape[1]; ++k) { // Loop over columns of the first and rows of the second
                result_data[i * other_shape[1] + j] += 
                    static_cast<ResultType>(lhs_data[i * shape[1] + k]) * 
                    static_cast<ResultType>(rhs_data[k * other_shape[1] + j]);
            }
        }
    }
}

Tensor Tensor::dot(const Tensor& other) const {
    checkDimensions(shape, other.get_shape());
    
    // Determine the result dtype based on both tensors
    Dtype result_dtype = promote_types(dtype, other.get_dtype());
    std::vector<int64_t> result_shape = {shape[0], other.shape[1]}; // m x p

    Tensor result(result_shape, result_dtype);

    // Use std::visit to handle the variant data for both tensors
    auto dotProductLambda = [&](const auto& lhs_data, const auto& rhs_data) {
        switch (result_dtype) {
            case Dtype::Float32: {
                std::vector<float> result_data(result_shape[0] * result_shape[1], 0.0f);
                compute_dot_product<float>(shape, other.get_shape(), lhs_data, rhs_data, result_data);
                result.set_data(result_data);
                break;
            }
            case Dtype::Float64: {
                std::vector<double> result_data(result_shape[0] * result_shape[1], 0.0);
                compute_dot_product<double>(shape, other.get_shape(), lhs_data, rhs_data, result_data);
                result.set_data(result_data);
                break;
            }
            case Dtype::Int32: {
                std::vector<int32_t> result_data(result_shape[0] * result_shape[1], 0);
                compute_dot_product<int32_t>(shape, other.get_shape(), lhs_data, rhs_data, result_data);
                result.set_data(result_data);
                break;
            }
            case Dtype::Int64: {
                std::vector<int64_t> result_data(result_shape[0] * result_shape[1], 0);
                compute_dot_product<int64_t>(shape, other.get_shape(), lhs_data, rhs_data, result_data);
                result.set_data(result_data);
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for dot product");
        }
    };

    std::visit(dotProductLambda, this->data, other.get_data());
    
    return result;
}

void Tensor::printRecursive(std::ostream& os, const std::vector<int64_t>& indices, size_t dim) const {
    if (dim == shape.size()) {
        // Reached the full depth of indices, print the value
        int64_t flatIndex = getFlatIndex(indices);
        std::visit([&os, flatIndex](const auto& dataVec) {
            os << std::fixed << std::setprecision(1) << dataVec[flatIndex];
        }, data);
    } else {
        os << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            std::vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            printRecursive(os, new_indices, dim + 1);

            if (i < shape[dim] - 1) {
                os << ", ";  // Comma between elements in the same dimension
            }
        }
        os << "]";
        
        // Ensure that the newline comes after the comma, not before
        if (dim == 1 && indices.size() < static_cast<size_t>(shape[0] - 1)) {
            os << ",\n        ";  // Add a newline and indentation after each row
        }
    }
}

// Prints the tensor in a structured format
std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "tensor(";
    
    // Print the tensor data
    printRecursive(oss, {}, 0);  // Use recursive method to print multi-dimensional arrays
    
    oss << ")";
    return oss.str();
}

// Implement the transpose method
Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::runtime_error("Transpose is only supported for 2D tensors.");
    }

    std::vector<int64_t> transposed_shape = {shape[1], shape[0]};
    Tensor result(transposed_shape, dtype);

    auto transposeLambda = [&](const auto& dataVec) {
        using DataType = std::decay_t<decltype(dataVec)>;
        DataType transposed_data(dataVec.size());

        for (int64_t i = 0; i < shape[0]; ++i) {
            for (int64_t j = 0; j < shape[1]; ++j) {
                transposed_data[j * shape[0] + i] = dataVec[i * shape[1] + j];
            }
        }
        result.set_data(transposed_data);
    };

    std::visit(transposeLambda, data);

    return result;
}

template <typename T>
py::array_t<T> Tensor::numpy_impl() const {
    std::vector<py::ssize_t> numpy_shape(shape.begin(), shape.end());

    return std::visit([&](const auto& dataVec) -> py::array_t<T> {
        using DataType = typename std::decay_t<decltype(dataVec)>::value_type;

        if constexpr (std::is_same_v<DataType, T>) {
            return py::array_t<T>(
                numpy_shape,
                {},                 // Let NumPy compute the strides
                dataVec.data(),     // Pointer to the data
                py::cast(this)      // Keep the Tensor alive
            );
        } else {
            throw std::runtime_error("Data type mismatch in numpy_impl()");
        }
    }, data);
}

py::array Tensor::numpy() const {
    switch (this->dtype) {
        case Dtype::Float32:
            return numpy_impl<float>();
        case Dtype::Float64:
            return numpy_impl<double>();
        case Dtype::Int32:
            return numpy_impl<int32_t>();
        case Dtype::Int64:
            return numpy_impl<int64_t>();
        default:
            throw std::runtime_error("Unsupported dtype in numpy()");
    }
}

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

void Tensor::ensureOnDevice() const {
    if (!clBuffer()) {
        auto& context = OpenCLManager::getInstance().getContext();

        // Variables to hold data pointer and size
        void* data_ptr = nullptr;
        size_t data_size = 0;

        // Use std::visit to handle the variant
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            // Check if arg is a vector
            static_assert(std::is_same_v<T, std::vector<float>> || 
                      std::is_same_v<T, std::vector<int64_t>> ||
                      std::is_same_v<T, std::vector<double>> ||
                      std::is_same_v<T, std::vector<int32_t>>,
                          "Unsupported data type");

            data_ptr = (void*)arg.data();  // Obtain data pointer

            // Calculate data size in bytes
            data_size = sizeof(typename T::value_type) * arg.size();

        }, data);

        // Create the OpenCL buffer
        cl_int err = CL_SUCCESS;
        clBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // Memory flags indicating read/write access and that the buffer should be initialized from host memory
                              data_size, data_ptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL buffer.");
        }
    }
}

void Tensor::readFromDevice() {
    auto& queue = OpenCLManager::getInstance().getQueue();

    void* data_ptr = nullptr;
    size_t data_size = 0;

    // Use std::visit to obtain data pointer and size
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        static_assert(std::is_same_v<T, std::vector<float>> || 
                      std::is_same_v<T, std::vector<int64_t>> ||
                      std::is_same_v<T, std::vector<double>> ||
                      std::is_same_v<T, std::vector<int32_t>>,
                      "Unsupported data type");

        data_ptr = (void*)arg.data();
        data_size = sizeof(typename T::value_type) * arg.size();

    }, data);

    // Enqueue read buffer command
    cl_int err = queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, data_size, data_ptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read from OpenCL buffer.");
    }
}

} // namespace mytorch