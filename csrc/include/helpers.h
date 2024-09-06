#pragma once

template <typename ele_type>
ostream& operator<<(ostream& os, const vector<ele_type>& vect_name) {
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

void checkDimensions(const vector<int64_t>& shape, const vector<int64_t>& other_shape) {
    if (shape[1] != other_shape[0]) {
        stringstream ss;
        ss << "Dimensions mismatch for dot product. " << shape[1] << " != " << other_shape[0];
        throw runtime_error(ss.str());
    }
}