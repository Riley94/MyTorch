#pragma once

template <typename ele_type>
std::ostream& operator<<(std::ostream& os, const std::vector<ele_type>& vect_name) {
    os << "[";
    for (size_t i = 0; i < vect_name.size(); ++i) {
        os << vect_name[i];
        if (i != vect_name.size() - 1) {
            os << " "; // Space between elements
        }
    }
    os << "]";
    return os; // Return the ostream object to allow chaining
}

void checkDimensions(const std::vector<int64_t>& shape, const std::vector<int64_t>& other_shape) {
    if (shape[1] != other_shape[0]) {
        std::stringstream ss;
        ss << "Dimensions mismatch for dot product. " << shape[1] << " != " << other_shape[0];
        throw std::runtime_error(ss.str());
    }
}