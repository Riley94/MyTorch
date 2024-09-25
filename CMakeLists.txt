cmake_minimum_required(VERSION 3.14)
project(MyTorch)

# GoogleTest requires at least C++14, but we will use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to your virtual environment
set(VENV_PATH "${CMAKE_SOURCE_DIR}")

# Determine the Python executable path based on the platform
if(WIN32)
    # Windows
    set(Python3_EXECUTABLE "${VENV_PATH}/Scripts/python.exe")
else()
    # Unix-like systems
    set(Python3_EXECUTABLE "${VENV_PATH}/bin/python")
endif()

set(Python3_INCLUDE_DIRS "${VENV_PATH}/include")


# Ensure that the Python interpreter exists
if(NOT EXISTS ${Python3_EXECUTABLE})
    message(FATAL_ERROR "Python interpreter not found at ${Python3_EXECUTABLE}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python version: ${Python3_VERSION}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")


# Add Tensor source files (modify path if necessary)
set(TENSOR_SRC csrc/MyTensor.cpp csrc/utils.cpp csrc/Proxy.cpp csrc/helpers.cpp)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch pybind11
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE) # Force pybind11 to use same Python as main project
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.5 # You can set this to any version
)
FetchContent_MakeAvailable(pybind11)

enable_testing()

add_executable(
  test_tensor
  csrc/tests/test_tensor.cc
  ${TENSOR_SRC}
)

# Add include directories for your project
target_include_directories(
  test_tensor PRIVATE
  ${CMAKE_SOURCE_DIR}/csrc/include   # Ensure this points to where MyTensor.h is located
  ${Python3_INCLUDE_DIRS}  # Include Python headers
)

target_link_libraries(
  test_tensor
  GTest::gtest_main
  pybind11::module
  ${Python3_LIBRARIES} # Python libraries
)

include(GoogleTest)
gtest_discover_tests(test_tensor)