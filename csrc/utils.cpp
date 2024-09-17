#include "utils.h"
#include "MyTensor.h"

int64_t numElements(const vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    int64_t totalSize = 1;
    for (int64_t s : shape) {
        totalSize *= s;
    }
    return totalSize;
}

Tensor ones_like(const Tensor& other, const Dtype& dtype) {
    return Tensor(other.get_shape(), vector<double>(other.size(), 1.0), dtype);
}

Tensor ones(const py::tuple& shape, const Dtype& dtype) {
    vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto item : shape)
    {
        temp.push_back(item.cast<int64_t>());
    }
    return ones_like(Tensor(temp), dtype);
}

Tensor zeros_like(const Tensor& other, const Dtype& dtype) {
    return Tensor(other.get_shape(), vector<double>(other.size(), 0.0), dtype);
}

Tensor zeros(const py::tuple& shape, const Dtype& dtype) {
    vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto item : shape)
    {
        temp.push_back(item.cast<int64_t>());
    }
    return zeros_like(Tensor(temp), dtype);
}

Tensor rand_like(const Tensor& other, const Dtype& dtype) {
    vector<double> random_data(other.size()); // Default to double, may change based on dtype

    random_device rd;
    mt19937 gen(rd()); // Mersenne Twister generator

    if (dtype == Dtype::Float32) {
        uniform_real_distribution<float> dis(0.0f, 1.0f); // Float distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = static_cast<double>(dis(gen)); // Cast float to double
        }
    } else if (dtype == Dtype::Float64) {
        uniform_real_distribution<double> dis(0.0, 1.0); // Double distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = dis(gen); // No cast needed for double
        }
    } else if (dtype == Dtype::Int32) {
        uniform_int_distribution<int32_t> dis(0, 100); // Int32 distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = static_cast<double>(dis(gen)); // Cast int to double
        }
    } else if (dtype == Dtype::Int64) {
        uniform_int_distribution<int64_t> dis(0, 100); // Int64 distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = static_cast<double>(dis(gen)); // Cast int to double
        }
    } else {
        throw invalid_argument("Unsupported dtype provided");
    }

    return Tensor(other.get_shape(), random_data, dtype);
}

Tensor rand(const py::tuple& shape, const Dtype& dtype) {
    vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto item : shape)
    {
        temp.push_back(item.cast<int64_t>());
    }
    return rand_like(Tensor(temp), dtype);
}

Tensor from_numpy(const py::array& np_array) {
    // Request buffer information from the NumPy array
    auto buffer_info = np_array.request();
    
    // Extract shape information from the buffer
    vector<int64_t> tensor_shape(buffer_info.shape.begin(), buffer_info.shape.end());

    if (py::format_descriptor<double>::format() == buffer_info.format) {
        // Create a vector to store the data from the NumPy array
        vector<double> tensor_data(static_cast<double*>(buffer_info.ptr), 
                                        static_cast<double*>(buffer_info.ptr) + buffer_info.size);
        return Tensor(tensor_shape, tensor_data, Dtype::Float64);
    }
    else if (py::format_descriptor<int64_t>::format() == buffer_info.format) {
        // Create a vector to store the data from the NumPy array
        vector<double> tensor_data(static_cast<int64_t*>(buffer_info.ptr), 
                                        static_cast<int64_t*>(buffer_info.ptr) + buffer_info.size);
        return Tensor(tensor_shape, tensor_data, Dtype::Int64);
    }
    else if (py::format_descriptor<int32_t>::format() == buffer_info.format) {
        // Create a vector to store the data from the NumPy array
        vector<double> tensor_data(static_cast<int32_t*>(buffer_info.ptr), 
                                        static_cast<int32_t*>(buffer_info.ptr) + buffer_info.size);
        return Tensor(tensor_shape, tensor_data, Dtype::Int32);
    }
    else if (py::format_descriptor<float>::format() == buffer_info.format) {
        // Create a vector to store the data from the NumPy array
        vector<double> tensor_data(static_cast<float*>(buffer_info.ptr), 
                                        static_cast<float*>(buffer_info.ptr) + buffer_info.size);
        return Tensor(tensor_shape, tensor_data, Dtype::Float32);
    }
    else {
        throw invalid_argument("Unsupported data type provided");
    }
}