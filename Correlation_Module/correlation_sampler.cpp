#include <torch/extension.h>

#include <vector>
#include <iostream>

// declarations

at::Tensor correlation_cuda_forward(
    at::Tensor input1,
    at::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

at::Tensor correlation_cpp_forward(
    at::Tensor input1,
    at::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

std::vector<at::Tensor> correlation_cuda_backward(
    at::Tensor grad_output,
    at::Tensor input1,
    at::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

std::vector<at::Tensor> correlation_cpp_backward(
    at::Tensor grad_output,
    at::Tensor input1,
    at::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor correlation_sample_forward(
    at::Tensor input1,
    at::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {
  if (input1.type().is_cuda()){
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    
    return correlation_cuda_forward(input1, input2, kH, kW, patchH, patchW,
                             padH, padW,
                             dilation_patchH, dilation_patchW,
                             dH, dW);
  }else{
    return correlation_cpp_forward(input1, input2, kH, kW, patchH, patchW,
                             padH, padW,
                             dilation_patchH, dilation_patchW,
                             dH, dW);
  }
}

std::vector<at::Tensor> correlation_sample_backward(
    at::Tensor input1,
    at::Tensor input2,
    at::Tensor grad_output,
    size_t kH, size_t kW,
    size_t patchH, size_t patchW,
    size_t padH, size_t padW,
    size_t dilation_patchH, size_t dilation_patchW,
    size_t dH, size_t dW) {

  if(grad_output.type().is_cuda()){
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    return correlation_cuda_backward(input1, input2, grad_output,
                              kH, kW, patchH, patchW,
                              padH, padW,
                              dilation_patchH, dilation_patchW,
                              dH, dW);
  }else{
    return correlation_cpp_backward(
                              input1, input2, grad_output,
                              kH, kW, patchH, patchW,
                              padH, padW,
                              dilation_patchH, dilation_patchW,
                              dH, dW);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_sample_forward, "Spatial Correlation Sampler Forward");
  m.def("backward", &correlation_sample_backward, "Spatial Correlation Sampler backward");
}
