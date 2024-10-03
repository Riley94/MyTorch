#include "MyTensor.h"
#include <gtest/gtest.h>
#include <pybind11/embed.h>  // Includes py::scoped_interpreter
#include "pybind_includes.h"

using namespace mytorch;

// The fixture for testing class Foo.
class TensorTest : public testing::Test {
protected:
    py::scoped_interpreter guard{}; // Initialize Python interpreter
  
    // Declare tensor1 and tensor2 as member variables.
    Tensor tensor1;
    Tensor tensor2;
    Tensor tensor3;

    // Constructor for setting up the test fixture.
    TensorTest() {
    }

    ~TensorTest() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override {
        // Code here will be called immediately after the constructor (right
        // before each test).
        tensor1 = Tensor(std::vector<int64_t>{2, 2}, std::vector<double>{1.0, 2.0, 3.0, 4.0}, Dtype::Float64);
        tensor2 = Tensor(std::vector<int64_t>{2, 2}, std::vector<double>{1.0, 1.0, 1.0, 1.0}, Dtype::Float64);
        tensor3 = Tensor(std::vector<int64_t>{3, 3}, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, Dtype::Float64);
    }

    void TearDown() override {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Class members declared here can be used by all tests in the test suite
    // for Foo.
};

std::vector<double> extractDataAsDouble(const Tensor& tensor) {
    std::vector<double> data;

    std::visit([&data](const auto& dataVec) {
        data.reserve(dataVec.size());
        for (const auto& val : dataVec) {
            data.push_back(static_cast<double>(val));
        }
    }, tensor.get_data());

    return data;
}

void checkData(const Tensor& result_tensor, 
               const std::vector<double>& expected_data, 
               const std::string& operation, 
               Dtype expected_dtype) {

    std::vector<std::string> dtype_names = {"Float32", "Float64", "Int32", "Int64"};
    // Check if the expected dtype matches the result tensor's dtype
    EXPECT_EQ(result_tensor.get_dtype(), expected_dtype) << "Dtype mismatch in " << operation 
                                                         << " result. Expected: " 
                                                         << dtype_names[static_cast<int>(expected_dtype)]
                                                         << ", but got: " 
                                                         << dtype_names[static_cast<int>(result_tensor.get_dtype())];

    // Extract actual data from the tensor using std::visit
    std::visit([&](const auto& dataVec) {
        using ActualDataType = typename std::decay_t<decltype(dataVec)>::value_type;

        // Check if the actual data type matches what we expect for the Dtype
        if constexpr (std::is_same_v<ActualDataType, double>) {
            EXPECT_EQ(expected_dtype, Dtype::Float64) << "Data type mismatch in " << operation;
        } else if constexpr (std::is_same_v<ActualDataType, float>) {
            EXPECT_EQ(expected_dtype, Dtype::Float32) << "Data type mismatch in " << operation;
        } else if constexpr (std::is_same_v<ActualDataType, int32_t>) {
            EXPECT_EQ(expected_dtype, Dtype::Int32) << "Data type mismatch in " << operation;
        } else if constexpr (std::is_same_v<ActualDataType, int64_t>) {
            EXPECT_EQ(expected_dtype, Dtype::Int64) << "Data type mismatch in " << operation;
        } else {
            FAIL() << "Unexpected data type in " << operation;
        }

        // Check the size of the data
        EXPECT_EQ(dataVec.size(), expected_data.size()) << "Size mismatch in " << operation << " result";

        // Check the values of the data
        for (size_t i = 0; i < expected_data.size(); ++i) {
            EXPECT_DOUBLE_EQ(static_cast<double>(dataVec[i]), expected_data[i]) << "Mismatch at index " << i << " for " << operation;
        }

    }, result_tensor.get_data());
}

TEST_F(TensorTest, Addition) {
    Tensor result = tensor1 + tensor2;
    Tensor resultScalar = tensor1 + 1.0;
    std::vector<double> expected_data = {2.0, 3.0, 4.0, 5.0};
    std::vector<double> expected_data_scalar = {2.0, 3.0, 4.0, 5.0};
    checkData(result, expected_data, "addition", Dtype::Float64);
    checkData(resultScalar, expected_data_scalar, "addition with scalar", Dtype::Float64);
}

TEST_F(TensorTest, Multiplication) {
    Tensor result = tensor1 * tensor2;
    Tensor resultScalar = tensor1 * 2.0;
    std::vector<double> expected_data = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> expected_data_scalar = {2.0, 4.0, 6.0, 8.0};
    checkData(result, expected_data, "multiplication", Dtype::Float64);
    checkData(resultScalar, expected_data_scalar, "multiplication with scalar", Dtype::Float64);
}

TEST_F(TensorTest, Subtraction) {
    Tensor result = tensor1 - tensor2;
    Tensor resultScalar = tensor1 - 1.0;
    std::vector<double> expected_data = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> expected_data_scalar = {0.0, 1.0, 2.0, 3.0};
    checkData(result, expected_data, "subtraction", Dtype::Float64);
    checkData(resultScalar, expected_data_scalar, "subtraction with scalar", Dtype::Float64);
}

TEST_F(TensorTest, Division) {
    Tensor result = tensor1 / tensor2;
    Tensor resultScalar = tensor1 / 1.0;
    Tensor resultDecimal = tensor1 / 3.0;
    std::vector<double> expected_data = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> expected_data_scalar = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> expected_data_decimal = {1.0/3.0, 2.0/3.0, 1.0, 4.0/3.0};
    checkData(result, expected_data, "division", Dtype::Float64);
    checkData(resultScalar, expected_data_scalar, "division with scalar", Dtype::Float64);
    checkData(resultDecimal, expected_data_decimal, "division with decimal output", Dtype::Float64);
}

// Test Tensor Negation
TEST_F(TensorTest, Negation) {
    Tensor result = -tensor1;
    std::vector<double> expected_data = {-1.0, -2.0, -3.0, -4.0};
    checkData(result, expected_data, "negation", Dtype::Float64);
}

// Test Tensor Dot Product
TEST_F(TensorTest, DotProduct) {
    Tensor result = tensor1.dot(tensor2);
    std::vector<double> expected_data = {3.0, 3.0, 7.0, 7.0};
    checkData(result, expected_data, "dot product", Dtype::Float64);
}

// Test Tensor Transpose
TEST_F(TensorTest, Transpose) {
    Tensor result = tensor1.transpose();
    std::vector<double> result_data = extractDataAsDouble(result);
    std::vector<double> expected_data = {1.0, 3.0, 2.0, 4.0};
    checkData(result, expected_data, "transpose", Dtype::Float64);
}

// Test Invalid Shape for Addition
TEST_F(TensorTest, InvalidShapeAddition) {
    
    // This should throw an assertion failure due to shape mismatch
    EXPECT_THROW(tensor1 + tensor3, std::runtime_error);
}

// Test in-place addition
TEST_F(TensorTest, InPlaceAddition) {
    Tensor tensor_to_change(std::vector<int64_t>{2, 2}, std::vector<double>{1.0, 1.0, 1.0, 1.0}, Dtype::Float64);
    tensor_to_change.add_(1.0);
    std::vector<double> expected_data = {2.0, 2.0, 2.0, 2.0};
    checkData(tensor_to_change, expected_data, "in-place addition", Dtype::Float64);
}

// Test Invalid Shape for Dot Product
TEST_F(TensorTest, InvalidShapeDotProduct) {
    // This should throw an assertion failure due to incompatible shapes for dot product
    EXPECT_THROW(tensor1.dot(tensor3), std::runtime_error);
}

// Test numpy conversion
TEST_F(TensorTest, NumpyConversion) {
    // Convert tensor1 to a numpy array
    auto npArray = tensor1.numpy();
    std::vector<double> tensor_data = extractDataAsDouble(tensor1);

    // Check if the numpy array has the same shape and data as tensor1
    py::buffer_info buffer = npArray.request();
    EXPECT_EQ(buffer.ndim, 2) << "Numpy array should have 2 dimensions";
    EXPECT_EQ(buffer.shape[0], 2) << "Numpy array should have 2 rows";
    EXPECT_EQ(buffer.shape[1], 2) << "Numpy array should have 2 columns";
    EXPECT_EQ(buffer.format, py::format_descriptor<double>::format()) << "Numpy array should have double data type";

    // Check if the data in the numpy array matches the data in tensor1
    double* data = static_cast<double*>(buffer.ptr);
    for (size_t i = 0; i < buffer.size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], tensor_data[i]) << "Mismatch in numpy array data at index " << i;
    }
}

TEST_F(TensorTest, GPUMath) {
    // Create tensors on the GPU
    Tensor tensorA({1000}, Dtype::Float64, DeviceType::GPU);
    Tensor tensorB({1000}, Dtype::Float64, DeviceType::GPU);

    // Initialize data
    std::vector<double> dataA(1000, 1.0);
    std::vector<double> dataB(1000, 2.0);
    tensorA.set_data(dataA);
    tensorB.set_data(dataB);

    // Perform addition on the GPU
    Tensor result = tensorA + tensorB;

    // Read result back to host if needed
    result.readFromDevice();

    std::vector<double> expectedData(1000, 3.0);
    checkData(result, expectedData, "addition on GPU", Dtype::Float64);
}

// Test Empty Tensor Initialization
/* TEST_F(TensorTest, TensorInit) {
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
} */

// GoogleTest entry point
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);  // Initialize GoogleTest
    return RUN_ALL_TESTS();                  // Run all test cases
}