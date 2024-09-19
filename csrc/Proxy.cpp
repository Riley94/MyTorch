#include "Proxy.h"

template <typename T>
Proxy& Proxy::operator=(const T& value) {
    std::visit([&value](auto& refVal) {
        refVal.get() = static_cast<typename std::decay_t<decltype(refVal.get())>>(value);
    }, ref);
    return *this;
}

Proxy::operator double() const {
    return std::visit([](const auto& refVal) -> double {
        return static_cast<double>(refVal.get());
    }, ref);
}