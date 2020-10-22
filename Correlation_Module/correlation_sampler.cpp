#include <torch/extension.h>

#include <vector>
#include <iostream>

// declarations

torch::Tensor correlation_cpp_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

std::vector<torch::Tensor> correlation_cpp_backward(
    torch::Tensor grad_output,
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

#ifdef USE_CUDA

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor correlation_cuda_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

std::vector<torch::Tensor> correlation_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

// C++ interface

torch::Tensor correlation_sample_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {
  if (input1.device().is_cuda()){
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    
    return correlation_cuda_forward(input1, input2, kH, kW, patchH, patchW,
                             padH, padW, dilationH, dilationW,
                             dilation_patchH, dilation_patchW,
                             dH, dW);
  }else{
    return correlation_cpp_forward(input1, input2, kH, kW, patchH, patchW,
                             padH, padW, dilationH, dilationW,
                             dilation_patchH, dilation_patchW,
                             dH, dW);
  }
}

std::vector<torch::Tensor> correlation_sample_backward(
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor grad_output,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {

  if(grad_output.device().is_cuda()){
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    return correlation_cuda_backward(input1, input2, grad_output,
                              kH, kW, patchH, patchW,
                              padH, padW,
                              dilationH, dilationW,
                              dilation_patchH, dilation_patchW,
                              dH, dW);
  }else{
    return correlation_cpp_backward(
                              input1, input2, grad_output,
                              kH, kW, patchH, patchW,
                              padH, padW,
                              dilationH, dilationW,
                              dilation_patchH, dilation_patchW,
                              dH, dW);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_sample_forward, "Spatial Correlation Sampler Forward");
  m.def("backward", &correlation_sample_backward, "Spatial Correlation Sampler backward");
}

#else

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_cpp_forward, "Spatial Correlation Sampler Forward");
  m.def("backward", &correlation_cpp_backward, "Spatial Correlation Sampler backward");
}

#endif
