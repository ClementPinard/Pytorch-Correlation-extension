# Correlation module

this is a custom C++/Cuda implementation of Correlation module, used e.g. in [FlowNetC](https://arxiv.org/abs/1504.06852)

This [tutorial](http://pytorch.org/tutorials/advanced/cpp_extension.html) was used as a basis for implementation, as well as
[NVIDIA's cuda code](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)

- Build and Install C++ and CUDA extensions by executing `python setup.py install`,
- Benchmark C++ vs. CUDA by running `python benchmark.py {cpu, cuda}`,
- Run gradient checks on the code by running `python grad_check.py --backend {cpu, cuda}`.

# Requirements

This module is expected to compile for Pytorch `0.4.1`, on `Python > 3.5` and `Python 2.7`.

# Installation

this module is available on pip

`pip install spatial-correlation-sampler`

# Usage

API has a few difference with NVIDIA's module
 * output is now a 5D tensor, which reflects the shifts horizontal and vertical.
 ```
input (B x C x H x W) -> output (B x PatchH x PatchW x oH x oW)
 ```
 * Output sizes `oH` and `oW` are no longer dependant of patch size, but only of kernel size and padding
 * Patch size `patch_size` is now the whole patch, and not only the radii.
 * `stride1` is now `stride` and`stride2` is `dilation_patch`, which behave like dilated convolutions
 * equivalent `max_displacement` is then `dilation_patch * (patch_size - 1) / 2`.
 * to get the right parameters for FlowNetC, you would have
 ```
kernel_size=1
patch_size=21,
stride=1,
padding=0,
dilation_patch=2
 ```

# Benchmark

 * default parameters are from `benchmark.py`, FlowNetC parameters are same as use in `FlowNetC` with a batch size of 4, described in [this paper](https://arxiv.org/abs/1504.06852), implemented [here](https://github.com/lmb-freiburg/flownet2) and [here](https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetC.py).
 * Feel free to file an issue to add entries to this with your hardware !

## CUDA Benchmark

 * See [here](https://gist.github.com/ClementPinard/270e910147119831014932f67fb1b5ea) for a benchmark script working with [NVIDIA](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)'s code, and Pytorch `0.3`.
 * Benchmark are launched with environment variable `CUDA_LAUNCH_BLOCKING` set to `1`.
 * Only `float32` is benchmarked.

 | implementation | Correlation parameters |  device |     pass |      min time |      avg time |
 | -------------- | ---------------------- | ------- | -------- | ------------: | ------------: |
 |           ours |                default | 980 GTX |  forward |  **5.313 ms** |  **5.339 ms** |
 |           ours |                default | 980 GTX | backward |    103.500 ms |    103.685 ms |
 |         NVIDIA |                default | 980 GTX |  forward |     12.763 ms |     12.844 ms |
 |         NVIDIA |                default | 980 GTX | backward | **74.043 ms** | **74.323 ms** |
 |                |                        |         |          |               |               |
 |           ours |               FlowNetC | 980 GTX |  forward |  **5.600 ms** |  **5.694 ms** |
 |           ours |               FlowNetC | 980 GTX | backward | **74.719 ms** | **75.122 ms** |
 |         NVIDIA |               FlowNetC | 980 GTX |  forward |      8.640 ms |      8.805 ms |
 |         NVIDIA |               FlowNetC | 980 GTX | backward |     75.757 ms |     76.873 ms |
 
### Notes
 * The large overhead of our implementation regarding `kernel_size` > 1 needs some investigation, feel free to
 dive in the code to improve it !
 * The backward pass of NVIDIA is not entirely correct when stride1 > 1 and kernel_size > 1, because not everything
 is computed, see [here](https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/correlation_package/src/correlation_cuda_kernel.cu#L120).

## CPU Benchmark

  * No other implementation is avalaible on CPU.

 | Correlation parameters |               device |     pass |    min time |    avg time |
 | ---------------------- | -------------------- | -------- | ----------: | ----------: |
 |                default | E5-2630 v3 @ 2.40GHz |  forward |  159.616 ms |  188.727 ms |
 |                default | E5-2630 v3 @ 2.40GHz | backward |  282.641 ms |  294.194 ms |
 |               FlowNetC | E5-2630 v3 @ 2.40GHz |  forward |  576.716 ms |  582.069 ms |
 |               FlowNetC | E5-2630 v3 @ 2.40GHz | backward | 1663.429 ms | 1663.429 ms |
