// OpenCLManager.cpp
#include "OpenCLManager.h"
#include <iostream>

OpenCLManager::OpenCLManager(mytorch::DeviceType deviceType, int deviceIndex) {
    try {
        // Discover platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        // Select the default platform (you can add logic to select a specific platform)
        cl::Platform platform = platforms[0];

        // Get devices for the platform based on the device type
        cl_device_type clDeviceType;
        switch (deviceType) {
            case mytorch::DeviceType::CPU:
                clDeviceType = CL_DEVICE_TYPE_CPU;
                break;
            case mytorch::DeviceType::GPU:
                clDeviceType = CL_DEVICE_TYPE_GPU;
                break;
            case mytorch::DeviceType::ALL:
                clDeviceType = CL_DEVICE_TYPE_ALL;
                break;
            default:
                clDeviceType = CL_DEVICE_TYPE_GPU;
        }

        std::vector<cl::Device> devices;
        platform.getDevices(clDeviceType, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found for the specified type.");
        }

        // Check if deviceIndex is within bounds
        if (deviceIndex < 0 || deviceIndex >= devices.size()) {
            throw std::out_of_range("Device index out of range.");
        }

        // Select the device based on deviceIndex
        device = devices[deviceIndex];

        // Create an OpenCL context and command queue
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);
    } catch (cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")\n";
        throw;
    }
}

// Static method to list available devices
std::vector<cl::Device> OpenCLManager::listDevices(mytorch::DeviceType deviceType) {
    std::vector<cl::Device> allDevices;
    try {
        // Discover platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Iterate over platforms
        for (const auto& platform : platforms) {
            // Get devices for the platform based on the device type
            cl_device_type clDeviceType;
            switch (deviceType) {
                case mytorch::DeviceType::CPU:
                    clDeviceType = CL_DEVICE_TYPE_CPU;
                    break;
                case mytorch::DeviceType::GPU:
                    clDeviceType = CL_DEVICE_TYPE_GPU;
                    break;
                case mytorch::DeviceType::ALL:
                    clDeviceType = CL_DEVICE_TYPE_ALL;
                    break;
                default:
                    clDeviceType = CL_DEVICE_TYPE_GPU;
            }

            std::vector<cl::Device> devices;
            platform.getDevices(clDeviceType, &devices);

            // Append devices to allDevices
            allDevices.insert(allDevices.end(), devices.begin(), devices.end());
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")\n";
    }

    return allDevices;
}