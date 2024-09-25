# Default target
all:
	python setup.py build_ext --inplace
	python -m build

# Clean target to remove generated files
clean:
	python setup.py clean --all
	rm -rf build/ dist/ *.egg-info

# Target for running tests
test:
	cmake -S . -B build
	cmake --build build
	cd build && ctest
	cd ..