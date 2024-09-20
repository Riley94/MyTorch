#include "helpers.h"

Dtype promote_types(const Dtype& dtype1, const Dtype& dtype2) {
    if (dtype1 == Dtype::Float64 || dtype2 == Dtype::Float64) {
        return Dtype::Float64;
    } else if (dtype1 == Dtype::Float32 || dtype2 == Dtype::Float32) {
        return Dtype::Float32;
    } else if (dtype1 == Dtype::Int64 || dtype2 == Dtype::Int64) {
        return Dtype::Int64;
    } else {
        return Dtype::Int32;
    }
}

void checkDimensions(const std::vector<int64_t>& shape, const std::vector<int64_t>& other_shape) {
    // 1. Check if the tensors are 2D
    if (shape.size() != 2 || other_shape.size() != 2) {
        throw std::runtime_error("Dot product is only defined for 2D tensors (matrices).");
    }

    // 2. Check if inner dimensions match
    if (shape[1] != other_shape[0]) {
        throw std::runtime_error("Dot product dimension mismatch: Columns of the first tensor must match rows of the second tensor.");
    }
}

int64_t numElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    int64_t totalSize = 1;
    for (int64_t s : shape) {
        totalSize *= s;
    }
    return totalSize;
}