#include "helpers.h"
#include "utils.h"
#include "MyTensor.h"

Tensor ones_like(const Tensor& other, const Dtype& dtype) {
    return Tensor(other.get_shape(), std::vector<double>(other.size(), 1.0), dtype);
}

Tensor ones(const py::tuple& shape, const Dtype& dtype) {
    std::vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto item : shape)
    {
        temp.push_back(item.cast<int64_t>());
    }
    return ones_like(Tensor(temp), dtype);
}

Tensor zeros_like(const Tensor& other, const Dtype& dtype) {
    return Tensor(other.get_shape(), std::vector<double>(other.size(), 0.0), dtype);
}

Tensor zeros(const py::tuple& shape, const Dtype& dtype) {
    std::vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto item : shape)
    {
        temp.push_back(item.cast<int64_t>());
    }
    return zeros_like(Tensor(temp), dtype);
}

Tensor rand_like(const Tensor& other, const Dtype& dtype) {
    std::vector<double> random_data(other.size()); // Default to double, may change based on dtype

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister generator

    if (dtype == Dtype::Float32) {
        std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Float distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = static_cast<double>(dis(gen)); // Cast float to double
        }
    } else if (dtype == Dtype::Float64) {
        std::uniform_real_distribution<double> dis(0.0, 1.0); // Double distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = dis(gen); // No cast needed for double
        }
    } else if (dtype == Dtype::Int32) {
        std::uniform_int_distribution<int32_t> dis(0, 100); // Int32 distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = static_cast<double>(dis(gen)); // Cast int to double
        }
    } else if (dtype == Dtype::Int64) {
        std::uniform_int_distribution<int64_t> dis(0, 100); // Int64 distribution
        for (int64_t i = 0; i < other.size(); ++i) {
            random_data[i] = static_cast<double>(dis(gen)); // Cast int to double
        }
    } else {
        throw std::invalid_argument("Unsupported dtype provided");
    }

    return Tensor(other.get_shape(), random_data, dtype);
}

Tensor rand(const py::tuple& shape, const Dtype& dtype) {
    std::vector<int64_t> temp;
    temp.reserve(shape.size());
    for (auto item : shape)
    {
        temp.push_back(item.cast<int64_t>());
    }
    return rand_like(Tensor(temp), dtype);
}

// given a numpy array and a dtype, create a tensor
template <typename T>
Tensor tensor_from_numpy(const py::array& np_array, Dtype dtype) {
    auto buffer_info = np_array.request();
    std::vector<int64_t> tensor_shape(buffer_info.shape.begin(), buffer_info.shape.end());
    size_t num_elements = buffer_info.size;
    
    std::vector<T> tensor_data(reinterpret_cast<T*>(buffer_info.ptr), // Use reinterpret_cast when casting from void* to another pointer type
                               reinterpret_cast<T*>(buffer_info.ptr) + num_elements);
    return Tensor(tensor_shape, tensor_data, dtype);
}

Tensor from_numpy(const py::array& np_array) {
    auto buffer_info = np_array.request();

    if (py::format_descriptor<double>::format() == buffer_info.format) {
        return tensor_from_numpy<double>(np_array, Dtype::Float64);
    } else if (py::format_descriptor<int64_t>::format() == buffer_info.format) {
        return tensor_from_numpy<int64_t>(np_array, Dtype::Int64);
    } else if (py::format_descriptor<int32_t>::format() == buffer_info.format) {
        return tensor_from_numpy<int32_t>(np_array, Dtype::Int32);
    } else if (py::format_descriptor<float>::format() == buffer_info.format) {
        return tensor_from_numpy<float>(np_array, Dtype::Float32);
    } else {
        throw std::invalid_argument("Unsupported data type provided");
    }
}