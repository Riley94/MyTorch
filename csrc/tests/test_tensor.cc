#include "MyTensor.h"
#include <gtest/gtest.h>

// The fixture for testing class Foo.
class TensorTest : public testing::Test {
 protected:
  // You can remove any or all of the following functions if their bodies would
  // be empty.

  TensorTest() {
     // You can do set-up work for each test here.
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
TEST(TensorTest, Addition) {
    Tensor tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor tensor2({2, 2}, {1.0, 1.0, 1.0, 1.0});
    Tensor result = tensor1 + tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {2.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Addition failed";
}

// Test Tensor Multiplication
TEST(TensorTest, Multiplication) {
    Tensor tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor tensor2({2, 2}, {5.0, 6.0, 7.0, 8.0});
    Tensor result = tensor1 * tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {5.0, 12.0, 21.0, 32.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Multiplication failed";
}

// Test Tensor Subtraction
TEST(TensorTest, Subtraction) {
    Tensor tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor tensor2({2, 2}, {1.0, 1.0, 1.0, 1.0});
    Tensor result = tensor1 - tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {0.0, 1.0, 2.0, 3.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Subtraction failed";
}

// Test Tensor Division
TEST(TensorTest, Division) {
    Tensor tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor tensor2({2, 2}, {1.0, 1.0, 1.0, 1.0});
    Tensor result = tensor1 / tensor2;

    // Check if result matches expected values
    vector<double> expected_data = {1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Division failed";
}

// Test Tensor Negation
TEST(TensorTest, Negation) {
    Tensor tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor result = -tensor1;

    // Check if result matches expected values
    vector<double> expected_data = {-1.0, -2.0, -3.0, -4.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Negation failed";
}

// Test Tensor Dot Product
TEST(TensorTest, DotProduct) {
    Tensor tensor1({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor tensor2({2, 2}, {5.0, 6.0, 7.0, 8.0});
    Tensor result = tensor1.dot(tensor2);

    // Check if result matches expected values
    vector<double> expected_data = {19.0, 22.0, 43.0, 50.0};
    EXPECT_EQ(result.get_data(), expected_data) << "Dot product failed";
}

// GoogleTest entry point
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);  // Initialize GoogleTest
    return RUN_ALL_TESTS();                  // Run all test cases
}