// OpenCLManager.h
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_TARGET_OPENCL_VERSION 300
#define CL_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <vector>
#include <mutex>
#include "DeviceType.h"

class OpenCLManager {
public:
    // Get the singleton instance of OpenCLManager with default device type (GPU)
    static OpenCLManager& getInstance(mytorch::DeviceType deviceType = mytorch::DeviceType::GPU, int deviceIndex = 0) {
        static OpenCLManager instance(deviceType, deviceIndex);
        return instance;
    }
    // Delete copy constructor and assignment operator to enforce singleton
    OpenCLManager(const OpenCLManager&) = delete;
    OpenCLManager& operator=(const OpenCLManager&) = delete;

    // Accessors for OpenCL resources
    cl::Context& getContext() { return context; }
    cl::Device& getDevice() { return device; }
    cl::CommandQueue& getQueue() { return queue; }

    // Method to list available devices
    static std::vector<cl::Device> listDevices(mytorch::DeviceType deviceType = mytorch::DeviceType::ALL);

private:
    // Constructor is private to enforce singleton
    OpenCLManager(mytorch::DeviceType deviceType, int deviceIndex);

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
};