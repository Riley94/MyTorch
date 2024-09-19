# Define variables
PYBIND_INCLUDES := $(shell python -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell python-config --extension-suffix)
PYTHON_LIB := $(shell python-config --ldflags)
PYTHON_INCLUDE := $(shell python-config --includes)

# Compiler and flags
CXX := g++
CXXFLAGS := -I csrc/include -O0 -g -Wall -shared -std=c++17 -Weffc++ -fPIC $(PYBIND_INCLUDES)

# Target
TARGET := MyTorchCPP$(PYTHON_EXT_SUFFIX)

# Source files
SOURCES = csrc/MyTensor.cpp csrc/MyTorch.cpp csrc/utils.cpp csrc/Proxy.cpp

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
	python setup.py clean --all
	rm MyTorchCPP*

# Target for running tests
test:
	cmake -S . -B build
	cmake --build build
	cd build && ctest
	cd ..

windows:
	python setup.py build_ext --inplace