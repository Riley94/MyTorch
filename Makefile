# Define variables
PYTHON := python3.11
PYTHON_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC $(PYTHON_INCLUDES)

# Target
TARGET := MyTensor$(PYTHON_EXT_SUFFIX)

# Source files
SRC := csrc/MyTensor.cpp

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Clean target to remove generated files
clean:
	rm -f $(TARGET)