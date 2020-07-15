#include <torch/types.h>
using namespace torch;

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

// Cuda tensor accessor definitions
// restrict pointer traits piroritize speed over memory consumption
#define TensorAcc4R PackedTensorAccessor<scalar_t,4,RestrictPtrTraits,int32_t>
#define TensorAcc5R PackedTensorAccessor<scalar_t,5,RestrictPtrTraits,int32_t>
#define TensorAcc6R PackedTensorAccessor<scalar_t,6,RestrictPtrTraits,int32_t>
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

#define THREADS_FORWARD 32
#define THREADS_BACKWARD 5


namespace {
template <typename scalar_t>
__global__ void correlation_cuda_forward_kernel(
    const TensorAcc4R rInput1,
    const TensorAcc4R rInput2,
    TensorAcc6R output,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {

  const int iH = rInput1.size(2);
  const int iW = rInput1.size(3);
  const int C = rInput1.size(1);
  const int W = output.size(5);

  const int n = blockIdx.x;
  const int c = blockIdx.y;
  const int h = blockIdx.z;
  const int thread = threadIdx.x;

  const int start_i = -padH + h * dH;

  const int patchRadH = dilation_patchH * (patchH - 1) / 2;
  const int patchRadW = dilation_patchW * (patchW - 1) / 2;
  //printf(" %d ",patchRadH )
  //printf(" %d ",patchRadW )

  for (int w=thread; w<W; w += THREADS_FORWARD){
    int start_j = -padW + w * dW;
    int i1 = start_i;
    int j1 = start_j;
    for(int ph = 0; ph < patchH; ++ph){
      int ph_dilated = ph * dilation_patchH - patchRadH;
      for(int pw = 0; pw < patchW; ++pw){
        int pw_dilated = pw * dilation_patchW - patchRadW;
        int i2 = i1 + ph_dilated;
        if WITHIN_BOUNDS(i1, i2, iH, iH){
          int j2 = j1 + pw_dilated;
          if WITHIN_BOUNDS(j1, j2, iW, iW){
            scalar_t v1 = rInput1[n][c][i1][j1];
            scalar_t v2 = rInput2[n][c][i2][j2];
            output[n][c][pw][ph][h][w] += v1 * v2;  // avoid permute
          }
        }
      }
    }
  }
}


template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input1(
    const TensorAcc6R gradOutput,
    const TensorAcc4R input2,
    TensorAcc4R gradInput1,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW,
    int batch) {
  const int iH = input2.size(2);
  const int iW = input2.size(3);

  const int H = gradOutput.size(4);
  const int W = gradOutput.size(5);
  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  
  const int n = batch;
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int ph_off = threadIdx.x;
  const int pw_off = threadIdx.y;

  const int h_2 = h + padH;
  const int w_2 = w + padW;
  const int start_i2 = h_2 / dH;
  const int start_j2 = w_2 / dW;
  /*we perform a module but since we have the quotient, we
  can cheat a bit*/
  const int h_off = h_2 - start_i2 * dH;
  const int w_off = w_2 - start_j2 * dW;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
  prod_sum[ph_off][pw_off] = 0;

  for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
    int i1 = h + dilation_patchH * (ph - patchRadH);
    for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
      int j1 = w + dilation_patchW * (pw - patchRadW);
      if WITHIN_BOUNDS(i1, j1, iH, iW) {
        scalar_t val = input2[n][c][i1][j1];
        for(int tmp1 = h_off, i = 0; tmp1 < kH; tmp1 += dH, ++i) {
          int i2 = start_i2 - i;
          for(int tmp2 = w_off, j = 0; tmp2 < kW; tmp2 += dW, ++j) {
            int j2 = start_j2 - j;
            if WITHIN_BOUNDS(i2, j2, H, W) {
              prod_sum[ph_off][pw_off] += gradOutput[n][c][pw][ph][i2][j2] * val;
            }
          }
        }
      }
    }
  }

  __syncthreads();

  if (ph_off == 0 && pw_off == 0){
    scalar_t reduce_sum =0;
    for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
      for (int pw = 0; pw < THREADS_BACKWARD; ++pw){
        reduce_sum += prod_sum[ph][pw];
      }
    }
    gradInput1[n][c][h][w] = reduce_sum;
  }
}


template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input2(
    const TensorAcc6R gradOutput,
    const TensorAcc4R input1,
    TensorAcc4R gradInput2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW,
    int batch) {
  const int iH = input1.size(2);
  const int iW = input1.size(3);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int H = gradOutput.size(4);
  const int W = gradOutput.size(5);
  
  const int n = batch;
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int ph_off = threadIdx.x;
  const int pw_off = threadIdx.y;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
  prod_sum[ph_off][pw_off] = 0;

  for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
    int i1 = h - dilation_patchH * (ph - patchRadH);
    for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
      int j1 = w - dilation_patchW * (pw - patchRadW);
      if WITHIN_BOUNDS(i1, j1, iH, iW) {
        scalar_t val = input1[n][c][i1][j1];
        
        const int h_2 = i1 + padH;
        const int w_2 = j1 + padW;
        const int start_i2 = h_2 / dH;
        const int start_j2 = w_2 / dW;
        const int h_off = h_2 - start_i2 * dH;
        const int w_off = w_2 - start_j2 * dW;
        
        for(int tmp1 = h_off, i = 0; tmp1 < kH; tmp1 += dH, ++i) {
          int i2 = start_i2 - i;
          for(int tmp2 = w_off, j = 0; tmp2 < kW; tmp2 += dW, ++j) {
            int j2 = start_j2 - j;
            if WITHIN_BOUNDS(i2, j2, H, W) {
              prod_sum[ph_off][pw_off] += gradOutput[n][c][pw][ph][i2][j2] * val;
            }
          }
        }
      }
    }
  }

  __syncthreads();

  if (ph_off == 0 && pw_off == 0){
    scalar_t reduce_sum =0;
    for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
      for (int pw = 0; pw < THREADS_BACKWARD; ++pw){
        reduce_sum += prod_sum[ph][pw];
      }
    }
    gradInput2[n][c][h][w] = reduce_sum;
  }
}
}


torch::Tensor correlation_cuda_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {

  const int batch_size = input1.size(0);
  const int nchannel = input1.size(1);
  const int iH = input1.size(2);
  const int iW = input1.size(3);

  const auto oH = (iH + 2 * padH - kH) / dH + 1;
  const auto oW = (iW + 2 * padW - kW) / dW + 1;

  auto output = at::zeros({batch_size, nchannel, patchW, patchH, oH, oW}, input1.options());
  auto trInput1 = input1;
  auto trInput2 = input2;
  //auto output = torch::zeros({batch_size, patchH, patchW, oH, oW}, input1.options());
  //auto trInput1 = input1.permute({0, 2, 3, 1}).contiguous();
  //auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();

  const int threads = THREADS_FORWARD;
  //const dim3 blocks(batch_size, oH, oW);
  const dim3 blocks(batch_size,nchannel,oH);
  //printf(" %d ",patchH );
  //printf(" %d ",patchW );

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "correlation_forward_cuda", ([&] {
    TensorAcc4R trInput1_acc  = trInput1.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R trInput2_acc = trInput2.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc6R output_acc = output.packed_accessor<scalar_t,6,RestrictPtrTraits,int32_t>();
    correlation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        trInput1_acc, trInput2_acc, output_acc,
        kH, kW, patchH, patchW, padH, padW,
        dilation_patchH, dilation_patchW, dH, dW);
  }));

  return output;
}

std::vector<torch::Tensor> correlation_cuda_backward(
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor gradOutput,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {
  
  auto gradInput1 = torch::zeros_like(input1);
  auto gradInput2 = torch::zeros_like(input2);

  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const int C = input1.size(1);

  const dim3 blocks(C, iH, iW);
  const dim3 threads(THREADS_BACKWARD, THREADS_BACKWARD);

  auto trInput1 = input1.contiguous();
  auto trInput2 = input2.contiguous();
  auto trgradOutput = gradOutput.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(trInput1.scalar_type(), "correlation_backward_cuda", ([&] {
    TensorAcc4R input1_acc = trInput1.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R input2_acc = trInput2.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R gradInput1_acc = gradInput1.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R gradInput2_acc = gradInput2.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc6R gradOutput_acc = trgradOutput.packed_accessor<scalar_t,6,RestrictPtrTraits,int32_t>();
//    TensorAcc5R gradOutput_acc = gradOutput.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();


    for (int n = 0; n < batch_size; ++n){
      correlation_cuda_backward_kernel_input1<scalar_t><<<blocks, threads>>>(
          gradOutput_acc, input2_acc, gradInput1_acc,
          kH, kW, patchH, patchW, padH, padW,
          dilation_patchH, dilation_patchW, dH, dW,
          n);
    }

    for (int n = 0; n < batch_size; ++n){
      correlation_cuda_backward_kernel_input2<scalar_t><<<blocks, threads>>>(
          gradOutput_acc, input1_acc, gradInput2_acc,
          kH, kW, patchH, patchW, padH, padW,
          dilation_patchH, dilation_patchW, dH, dW,
          n);
    }
  }
));

  return {gradInput1, gradInput2};
}
