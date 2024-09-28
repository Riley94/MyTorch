#include <CL/cl.h>

#include "helpers.h"
#include "utils.h"
#include "MyTensor.h"
#include "pybind_includes.h"

namespace mytorch {

Tensor ones_like(const Tensor& other, const Dtype& dtype) {
    switch (dtype)
    {
    case Dtype::Float32:
        return Tensor(other.get_shape(), std::vector<float>(other.size(), 1.0), dtype);
        break;
    case Dtype::Float64:
        return Tensor(other.get_shape(), std::vector<double>(other.size(), 1.0), dtype);
        break;
    case Dtype::Int32:
        return Tensor(other.get_shape(), std::vector<int32_t>(other.size(), 1), dtype);
        break;
    case Dtype::Int64:
        return Tensor(other.get_shape(), std::vector<int64_t>(other.size(), 1), dtype);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype provided");
        break;
    }
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
    switch (dtype)
    {
    case Dtype::Float32:
        return Tensor(other.get_shape(), std::vector<float>(other.size(), 0.0), dtype);
        break;
    case Dtype::Float64:
        return Tensor(other.get_shape(), std::vector<double>(other.size(), 0.0), dtype);
        break;
    case Dtype::Int32:
        return Tensor(other.get_shape(), std::vector<int32_t>(other.size(), 0), dtype);
        break;
    case Dtype::Int64:
        return Tensor(other.get_shape(), std::vector<int64_t>(other.size(), 0), dtype);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype provided");
        break;
    }
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

template <typename T>
Tensor generate_random_data(const Tensor& other, const Dtype& dtype, std::mt19937& gen) {
    std::vector<T> random_data(other.size());
    std::uniform_real_distribution<double> dis(0.0, 1.0); // double distribution
    for (int64_t i = 0; i < other.size(); ++i) {
        random_data[i] = static_cast<T>(dis(gen)); // Ensure type T
    }
    return Tensor(other.get_shape(), random_data, dtype);
}

Tensor rand_like(const Tensor& other, const Dtype& dtype) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister generator

    switch (dtype)
    {
    case Dtype::Float32:
        return generate_random_data<float>(other, dtype, gen);
        break;
    case Dtype::Float64:
        return generate_random_data<double>(other, dtype, gen);
        break;
    case Dtype::Int32:
        return generate_random_data<int32_t>(other, dtype, gen);
        break;
    case Dtype::Int64:
        return generate_random_data<int64_t>(other, dtype, gen);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype provided");
        break;
    }
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

    if (buffer_info.format == py::format_descriptor<double>::format()) {
        return tensor_from_numpy<double>(np_array.cast<py::array_t<double>>(), Dtype::Float64);
    } else if (buffer_info.format == py::format_descriptor<int64_t>::format()) {
        return tensor_from_numpy<int64_t>(np_array.cast<py::array_t<int64_t>>(), Dtype::Int64);
    } else if (buffer_info.format == py::format_descriptor<int32_t>::format()) {
        return tensor_from_numpy<int32_t>(np_array.cast<py::array_t<int32_t>>(), Dtype::Int32);
    } else if (buffer_info.format == py::format_descriptor<float>::format()) {
        return tensor_from_numpy<float>(np_array.cast<py::array_t<float>>(), Dtype::Float32);
    } else if (buffer_info.format == "l") { // explicitly check for 'l' format, present on some linux systems.
        if (sizeof(long) == 8) {
            return tensor_from_numpy<int64_t>(np_array.cast<py::array_t<int64_t>>(), Dtype::Int64);
        } else if (sizeof(long) == 4) {
            return tensor_from_numpy<int32_t>(np_array.cast<py::array_t<int32_t>>(), Dtype::Int32);
        } else {
            throw std::runtime_error("Unsupported size for 'long' type");
        }
    } else {
        throw std::invalid_argument("Unsupported data type provided. Got: " + std::string(buffer_info.format)
                                    + ". Expected: " + std::string(py::format_descriptor<double>::format()) 
                                    + ", " + std::string(py::format_descriptor<int64_t>::format()) 
                                    + ", " + std::string(py::format_descriptor<int32_t>::format()) + ", " 
                                    + std::string(py::format_descriptor<float>::format()) + ", or 'l'");
    }
}

void get_devices() {
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    py::print("Found", platformCount, "OpenCL platform(s).\n");

    for (cl_uint i = 0; i < platformCount; ++i) {
        char platformName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
        py::print("Platform", i + 1, ":", platformName);

        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        py::print("  Found", deviceCount, "device(s).");

        for (cl_uint j = 0; j < deviceCount; ++j) {
            char deviceName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            py::print("  Device", j + 1, ":", deviceName);
        }
        py::print();  // Print an empty line for spacing
    }
    return;
}

} // namespace mytorch