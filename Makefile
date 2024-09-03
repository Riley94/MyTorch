# Define variables
PYTHON := python3.11
PYTHON_INCLUDE := -IC:\msys64\ucrt64\include\python3.11
PYTHON_LIB := -LC:\msys64\ucrt64\lib\python3.11
PYTHON_CFLAGS := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

# Manually set include paths if the above doesn't work
EXTRA_INCLUDES := -IC:\msys64\ucrt64\lib\python3.11\site-packages\pybind11\include

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC $(PYTHON_INCLUDE) $(PYTHON_CFLAGS) $(EXTRA_INCLUDES)

# Target
TARGET := MyTensor$(PYTHON_EXT_SUFFIX)

# Source files
SRC := MyTensor.cpp

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(PYTHON_LIB) -lpython3.11

# Clean target to remove generated files
clean:
	rm -f $(TARGET)