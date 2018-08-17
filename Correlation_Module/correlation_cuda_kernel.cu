#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THC.h>
#include <THC/THCDeviceTensor.cuh>


#include <vector>
#include <iostream>

#define dTensor4R THCDeviceTensor<scalar_t, 4, size_t, RestrictPtrTraits>
#define dTensor5R THCDeviceTensor<scalar_t, 5, size_t, RestrictPtrTraits>
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

#define THREADS_PER_BLOCK 32

template <typename scalar_t, int dims>
THCDeviceTensor<scalar_t, dims, size_t, RestrictPtrTraits>
toDeviceTensor(at::Tensor t) {
  return THCDeviceTensor<scalar_t, dims, size_t, RestrictPtrTraits>
  (t.data<scalar_t>(), (size_t*) t.sizes().data(), (size_t*) t.strides().data());
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


using namespace at;
namespace {
template <typename scalar_t>
__global__ void correlation_cuda_forward_kernel(
    const dTensor4R rInput1,
    const dTensor4R rInput2,
    dTensor5R output,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {
  
  const int C = rInput1.getSize(3);
  const int iH = rInput1.getSize(1);
  const int iW = rInput1.getSize(2);

  const int H = output.getSize(3);
  const int W = output.getSize(4);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;

  __shared__ scalar_t prod_sum[THREADS_PER_BLOCK];

  for(int ph = 0; ph < patchH; ph++){
    for(int pw = 0; pw < patchW; pw++){
      prod_sum[threadIdx.x] = 0;
      for (int i=0; i<kH; ++i){
        int i1 = -padH + h * dH + i;
        int i2 = i1 - (ph - patchRadH)  * dilation_patchH;
        if WITHIN_BOUNDS(i1, i2, iH, iH){
          for (int j=0; j<kW; ++j){
            int j1 = -padW + w * dW + j;
            int j2 = j1 - (pw - patchRadW) * dilation_patchW;
            if WITHIN_BOUNDS(j1, j2, iW, iW){
              for (int c=threadIdx.x; c<C; c += blockDim.x){
                scalar_t v1 = rInput1[n][i1][j1][c];
                scalar_t v2 = rInput2[n][i2][j2][c];
                prod_sum[threadIdx.x] += v1 * v2;
              }
            }
          }
        }
      }
      // accumulate 
      __syncthreads();
      if (threadIdx.x == 0) {
        scalar_t reduce_sum = 0;
        for (int index = 0; index < blockDim.x; ++index) {
          reduce_sum += prod_sum[index];
        }
        output[n][ph][pw][h][w] = reduce_sum;
      }
    }
  }
}


template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input1(
    const dTensor5R gradOutput,
    const dTensor4R input2,
    dTensor4R gradInput1,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW,
    int nElem) {
  const int C = input2.getSize(1);
  const int iH = input2.getSize(2);
  const int iW = input2.getSize(3);

  const int H = gradOutput.getSize(3);
  const int W = gradOutput.getSize(4);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  
  const int n = blockIdx.y;

  int c,h,w;
  CUDA_KERNEL_LOOP(index, nElem){
    w = index % iW;
    h = (index / iW) % iH;
    c = (index / (iW * iH)) % C;
    for (int ph = 0; ph < patchH; ++ph) {
      int i1 = h - dilation_patchH * (ph - patchRadH);
      for (int pw = 0; pw < patchW; ++pw) {
        int j1 = w - dilation_patchW * (pw - patchRadW);
        if WITHIN_BOUNDS(i1, j1, iH, iW) {
          scalar_t val = input2[n][c][i1][j1];
          for(int i = 0; i < kH; i += dH) {
            int i2 = (h + padH - i) / dH;
            for(int j = 0; j < kW; j += dW) {
              int j2 = (w + padW - j) / dW;
              if WITHIN_BOUNDS(i2, j2, H, W) {
                gradInput1[n][c][h][w] += gradOutput[n][ph][pw][i2][j2] * val;
              }
            }
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input2(
    const dTensor5R gradOutput,
    const dTensor4R input1,
    dTensor4R gradInput2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW,
    int nElem) {
  const int C = input1.getSize(1);
  const int iH = input1.getSize(2);
  const int iW = input1.getSize(3);

  const int H = gradOutput.getSize(3);
  const int W = gradOutput.getSize(4);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  
  const int n = blockIdx.y;

  int c,h,w;
  CUDA_KERNEL_LOOP(index, nElem){
    w = index % iW;
    h = (index / iW) % iH;
    c = (index / (iW * iH)) % C;
    for (int ph = 0; ph < patchH; ++ph) {
      int i1 = h - dilation_patchH * (ph - patchRadH);
      for (int pw = 0; pw < patchW; ++pw) {
        int j1 = w - dilation_patchW * (pw - patchRadW);
        if WITHIN_BOUNDS(i1, j1, iH, iW) {
          scalar_t val = input1[n][c][i1][j1];
          for(int i = 0; i < kH; i += dH) {
            int i2 = (i1 + padH - i) / dH;
            for(int j = 0; j < kW; j += dW) {
              int j2 = (j1 + padW - j) / dW;
              if WITHIN_BOUNDS(i2, j2, H, W) {
                gradInput2[n][c][h][w] += gradOutput[n][ph][pw][i2][j2] * val;
              }
            }
          }
        }
      }
    }
  }
}
}

at::Tensor correlation_cuda_forward(
    at::Tensor input1,
    at::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {
  
  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const at::IntList output_size = {
    batch_size,
    patchH,
    patchW,
    (iH + 2 * padH - kH) / dH + 1,
    (iW + 2 * padW - kW) / dW + 1
  };

  auto output = at::zeros(output_size, input1.options());
  
  auto trInput1 = input1.permute({0, 2, 3, 1}).contiguous();
  auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();
  
  const int threads = THREADS_PER_BLOCK;
  const dim3 blocks(batch_size, output_size[3], output_size[4]);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "correlation_forward_cuda", ([&] {
    correlation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<scalar_t,4>(trInput1),
        toDeviceTensor<scalar_t,4>(trInput2),
        toDeviceTensor<scalar_t,5>(output),
        kH, kW, patchH, patchW, padH, padW,
        dilation_patchH, dilation_patchW, dH, dW);
  }));

  return output;
}

std::vector<at::Tensor> correlation_cuda_backward(
    at::Tensor input1,
    at::Tensor input2,
    at::Tensor gradOutput,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {
  
  auto gradInput1 = at::zeros_like(input1);
  auto gradInput2 = at::zeros_like(input2);

  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const int C = input1.size(1);

  const int nElem = C * iH * iW;

  const int threads = 1024;
  const dim3 blocks((nElem + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "correlation_backward_cuda", ([&] {
    correlation_cuda_backward_kernel_input1<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<scalar_t,5>(gradOutput),
        toDeviceTensor<scalar_t,4>(input2),
        toDeviceTensor<scalar_t,4>(gradInput1),
        kH, kW, patchH, patchW, padH, padW,
        dilation_patchH, dilation_patchW, dH, dW,
        nElem);
    correlation_cuda_backward_kernel_input2<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<scalar_t,5>(gradOutput),
        toDeviceTensor<scalar_t,4>(input1),
        toDeviceTensor<scalar_t,4>(gradInput2),
        kH, kW, patchH, patchW, padH, padW,
        dilation_patchH, dilation_patchW, dH, dW,
        nElem);
  }));

  return {gradInput1, gradInput2};
}
