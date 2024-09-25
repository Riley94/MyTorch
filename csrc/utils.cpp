#include "helpers.h"
#include "utils.h"
#include "MyTensor.h"
#include "pybind_includes.h"

namespace mytorch {

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

template <typename T>
Tensor tensor_from_numpy(const py::array_t<T>& np_array, Dtype dtype) {
    auto buffer_info = np_array.request();
    std::vector<int64_t> shape;
    shape.reserve(buffer_info.ndim);
    for (int i = 0; i < buffer_info.ndim; ++i) {
        shape.push_back(buffer_info.shape[i]);
    }

    std::vector<T> tensor_data(reinterpret_cast<T*>(buffer_info.ptr),
                               reinterpret_cast<T*>(buffer_info.ptr) + buffer_info.size);

    return Tensor(shape, tensor_data, dtype);
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
        throw std::invalid_argument("Unsupported data type provided. Got: " + std::string(buffer_info.format));
    }
}

} // namespace mytorch