#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list
sudo apt update

# Ensure that the software-properties-common package is installed to be able to manage PPAs
sudo apt install software-properties-common

# common repository for alternate python versions
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.12, development libraries, and venv
sudo apt install -y python3.12 python3.12-dev python3.12-venv

# Install CMake
sudo apt install -y cmake

# Install GoogleTest and move to the appropriate directory
sudo apt install -y libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo mv lib/*.a /usr/lib/

# Go back to the original directory ~/MyTorch
cd ~/MyTorch

# Create a Python 3.12 virtual environment
python3.12 -m venv .

# Output a message to the user
echo "Python 3.12 environment is ready."
echo "To activate the virtual environment, run:"
echo "source ./bin/activate"