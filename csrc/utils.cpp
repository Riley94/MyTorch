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
Tensor from_numpy(const py::array_t<T>& np_array) {
    auto buffer_info = np_array.request();

    Dtype dtype = getDtypeFromCppType<T>();

    std::vector<int64_t> tensor_shape(buffer_info.shape.begin(), buffer_info.shape.end());
    size_t num_elements = buffer_info.size;
    
    std::vector<T> tensor_data(reinterpret_cast<T*>(buffer_info.ptr), // Use reinterpret_cast when casting from void* to another pointer type
                               reinterpret_cast<T*>(buffer_info.ptr) + num_elements);
    return Tensor(tensor_shape, tensor_data, dtype);
}

// Explicit template instantiation
template Tensor from_numpy<double>(const py::array_t<double>& np_array);
template Tensor from_numpy<float>(const py::array_t<float>& np_array);
template Tensor from_numpy<int64_t>(const py::array_t<int64_t>& np_array);
template Tensor from_numpy<int32_t>(const py::array_t<int32_t>& np_array);

} // namespace mytorch