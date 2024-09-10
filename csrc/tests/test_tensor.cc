#include "MyTensor.h"
#include <gtest/gtest.h>
#include <pybind11/embed.h>  // Includes py::scoped_interpreter

// The fixture for testing class Foo.
class TensorTest : public testing::Test {
 protected:
  // You can remove any or all of the following functions if their bodies would
  // be empty.

  // Declare tensor1 and tensor2 as member variables.
  Tensor tensor1;
  Tensor tensor2;
  Tensor tensor3;

  // Constructor for setting up the test fixture.
  TensorTest() : tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0}),
                 tensor2({2, 2}, {1.0, 1.0, 1.0, 1.0}),
                 tensor3({3, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}) {
    // You can do additional set-up work for each test here.
  }

  ~TensorTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

// Example test case: Test Tensor Addition
TEST_F(TensorTest, Addition) {
    Tensor result = tensor1 + tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {2.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Addition failed";
}

// Test Tensor Multiplication
TEST_F(TensorTest, Multiplication) {
    Tensor result = tensor1 * tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Multiplication failed";

    // Scalar Multiplication
    double scalar = 2.0;
    result = tensor1 * scalar;  // Assuming you have an overloaded operator for scalar multiplication

    // Check if result matches expected values
    expected_data = {2.0, 4.0, 6.0, 8.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Scalar multiplication failed";
}

// Test Tensor Subtraction
TEST_F(TensorTest, Subtraction) {
    Tensor result = tensor1 - tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {0.0, 1.0, 2.0, 3.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Subtraction failed";
}

// Test Tensor Division
TEST_F(TensorTest, Division) {
    Tensor result = tensor1 / tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Division failed";
}

// Test Tensor Negation
TEST_F(TensorTest, Negation) {
    Tensor result = -tensor1;

    // Check if result matches expected values
    vector<double> expected_data = {-1.0, -2.0, -3.0, -4.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Negation failed";
}

// Test Tensor Dot Product
TEST_F(TensorTest, DotProduct) {
    Tensor result = tensor1.dot(tensor2);

    // Check if result matches expected values
    vector<double> expected_data = {3.0, 3.0, 7.0, 7.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Dot product failed";
}

// Test Tensor Transpose
TEST_F(TensorTest, Transpose) {
    Tensor result = tensor1.transpose();

    // Check if result matches expected values
    vector<double> expected_data = {1.0, 3.0, 2.0, 4.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Transpose failed";
}

// Test Invalid Shape for Addition
TEST_F(TensorTest, InvalidShapeAddition) {
    
    // This should throw an assertion failure due to shape mismatch
    EXPECT_THROW(tensor1 + tensor3, runtime_error);
}

// Test Invalid Shape for Dot Product
TEST_F(TensorTest, InvalidShapeDotProduct) {
    // This should throw an assertion failure due to incompatible shapes for dot product
    EXPECT_THROW(tensor1.dot(tensor3), runtime_error);
}

// Test Empty Tensor Initialization
TEST_F(TensorTest, TensorInit) {
    py::scoped_interpreter guard{}; // Initialize Python interpreter

    Tensor tensorEmptyArr({}, {}); // Empty init

    // Check if the data is empty and shape is empty
    EXPECT_EQ(tensorEmptyArr.get_data().size(), 0) << "Array initialization failed. Empty tensor data size should be 0";
    EXPECT_EQ(tensorEmptyArr.get_shape().size(), 0) << "Array initialization failed. Empty tensor shape size should be 0";

    py::list listEmpty = py::cast(vector<double>{});
    Tensor tensorEmptyList(listEmpty);

    // Check if the data is empty and shape is empty
    EXPECT_EQ(tensorEmptyList.get_data().size(), 0) << "Python list initialization failed. Empty tensor data size should be 0";
    EXPECT_EQ(tensorEmptyList.get_shape().size(), 0) << "Python list initialization failed. Empty tensor shape size should be 0";

    Tensor tensor1DArr({1}, {1.0}); // 1D with array init
    const vector<double> expected1D{1.0};

    // Check if the data is empty and shape is empty
    EXPECT_EQ(tensor1DArr.get_data().size(), 1) << "Array initialization failed. 1-D tensor data size should be 1";
    EXPECT_EQ(tensor1DArr.get_shape().size(), 1) << "Array initialization failed. 1-D tensor shape size should be 1";
    // Check if the data is empty and shape is empty
    EXPECT_EQ(tensor1DArr.get_data(), expected1D) << "Array initialization failed. 1-D tensor data should be 1.0";
    EXPECT_EQ(tensor1DArr.get_shape()[0], expected1D.size()) << "Array initialization failed. 1-D tensor shape should be 1";

    py::list list1D = py::cast(vector<double>{1.0});
    Tensor tensorList1D(list1D);

    // Check if the data is empty and shape is empty
    EXPECT_EQ(tensorList1D.get_data().size(), 1) << "Python list initialization failed. 1-D tensor data size should be 1";
    EXPECT_EQ(tensorList1D.get_shape().size(), 1) << "Python list initialization failed. 1-D tensor shape size should be 1";
    // Check if the data is empty and shape is empty
    EXPECT_EQ(tensorList1D.get_data(), expected1D) << "Python list initialization failed. 1-D tensor data should be 1.0";
    EXPECT_EQ(tensorList1D.get_shape()[0], expected1D.size()) << "Python list initialization failed. 1-D tensor shape should be 1";
}

// GoogleTest entry point
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);  // Initialize GoogleTest
    return RUN_ALL_TESTS();                  // Run all test cases
}