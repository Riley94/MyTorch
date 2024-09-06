#pragma once

#include <random>

Tensor ones_like(const Tensor& other) {
    return Tensor(other.get_shape(), vector<double>(other.size(), 1.0));
}

Tensor zeros_like(const Tensor& other) {
    return Tensor(other.get_shape());
}

Tensor rand_like(const Tensor& other) {
    vector<double> random_data(other.size());

    // Create a random number generator
    random_device rd;  // Seed for randomness
    mt19937 gen(rd()); // Mersenne Twister generator
    uniform_real_distribution<double> dis(0.0, 1.0); // Uniform distribution between 0 and 1

    for (int64_t i = 0; i < other.size(); ++i) {
        random_data[i] = dis(gen); // Generate random number
    }

    return Tensor(other.get_shape(), random_data);
}