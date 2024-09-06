# Define variables
PYTHON := python3.11
PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)
PYTHON_LIB := $(shell $(PYTHON)-config --ldflags)
PYTHON_INCLUDE := $(shell $(PYTHON)-config --includes)

# Compiler and flags
CXX := g++
CXXFLAGS := -I csrc/include -O0 -g -Wall -shared -std=c++17 -fPIC $(PYBIND_INCLUDES)

# Target
TARGET := MyTorch$(PYTHON_EXT_SUFFIX)

# Source files
SOURCES = csrc/MyTensor.cpp csrc/MyTorch.cpp

# Object files
OBJECTS := $(patsubst csrc/%.cpp,csrc/build/%.o,$(SOURCES))

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

# Compile individual source files into object files
csrc/build/%.o: csrc/%.cpp | csrc/build
	$(CXX) $(CXXFLAGS) -c $< -o $@

csrc/bin/%.bin: csrc/build/%.o
	$(CXX) $(CXXFLAGS) $< -o $@

# Create the bin directory if it doesn't exist
csrc/bin:
	mkdir -p csrc/bin

# Create the build directory if it doesn't exist
csrc/build:
	mkdir -p csrc/build

# Create the tests directory if it doesn't exist
csrc/tests:
	mkdir -p csrc/tests

# Clean target to remove generated files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Target for running tests
test:
	cmake -S . -B build
	cmake --build build
	cd build && ctest
	cd ..