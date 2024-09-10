#pragma once

#include <random>

enum class Dtype {
    Float32,
    Float64,
    Int32,
    Int64
};

Tensor ones_like(const Tensor& other) {
    return Tensor(other.get_shape(), vector<double>(other.size(), 1.0));
}

Tensor zeros_like(const Tensor& other) {
    return Tensor(other.get_shape());
}

Tensor rand_like(const Tensor& other, Dtype dtype = Dtype::Float64) {
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

    return Tensor(other.get_shape(), random_data);
}