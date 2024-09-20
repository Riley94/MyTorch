#pragma once

#include <variant>
#include <functional>
#include <cstdint>
#include <type_traits>

class Proxy {
private:
    std::variant<
        std::reference_wrapper<double>,
        std::reference_wrapper<float>,
        std::reference_wrapper<int32_t>,
        std::reference_wrapper<int64_t>
    > ref;

public:
    Proxy(double& val) : ref(std::ref(val)) {}
    Proxy(float& val) : ref(std::ref(val)) {}
    Proxy(int32_t& val) : ref(std::ref(val)) {}
    Proxy(int64_t& val) : ref(std::ref(val)) {}

    // Assignment operator
    template <typename T>
    Proxy& operator=(const T& value);

    // Conversion operator for double
    operator double() const;
};