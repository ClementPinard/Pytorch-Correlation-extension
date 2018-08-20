# Correlation module

this is a custom C++/Cuda implementation of Correlation module, used e.g. in [FlowNetC](https://arxiv.org/abs/1504.06852)

This [tutorial](http://pytorch.org/tutorials/advanced/cpp_extension.html) was used as a basis for implementation.

- Build and Install C++ and CUDA extensions by executing `python setup.py install`,
- Benchmark C++ vs. CUDA by running `python benchmark.py {cpu, cuda}`,
- Run gradient checks on the code by running `python grad_check.py --backend {cpu, cuda}`.

# Requirements

This module is expected to compile for Pytorch `0.4.1`, on `Python > 3.5` and `Python 2.7`.

# Installation

this module is available on pip

`pip install spatial-correlation-sampler`

# Benchmark

 * default parameters are from `benchmark.py`, FlowNetC parameters are same as use in `FlowNetC` with a batch size of 4, described in [this paper](https://arxiv.org/abs/1504.06852), implemented [here](https://github.com/lmb-freiburg/flownet2) and [here](https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetC.py).
 * Feel free to file an issue to add entries to this with your hardware !

## CUDA Benchmark

 * See [here](https://gist.github.com/ClementPinard/270e910147119831014932f67fb1b5ea) for a benchmark script working with [NVIDIA](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)'s code, and Pytorch `0.3`.
 * Benchmark are launched with environment variable `CUDA_LAUNCH_BLOCKING` set to `1`.
 * Only `float32` is benchmarked.

 | implementation | Correlation parameters |  device |     pass |   min time |   avg time |
 | -------------- | ---------------------- | ------- | -------- | ---------: | ---------: |
 |           ours |                default | 980 GTX |  forward |  24.912 ms |  25.202 ms |
 |           ours |                default | 980 GTX | backward | 148.341 ms | 148.827 ms |
 |         NVIDIA |                default | 980 GTX |  forward |  23.680 ms |  23.797 ms |
 |         NVIDIA |                default | 980 GTX | backward | 118.519 ms | 119.367 ms |
 |                |                        |         |          |            |            |
 |           ours |               FlowNetC | 980 GTX |  forward |  10.132 ms |  10.273 ms |
 |           ours |               FlowNetC | 980 GTX | backward | 116.646 ms | 117.131 ms |
 |         NVIDIA |               FlowNetC | 980 GTX |  forward |   8.640 ms |   8.805 ms |
 |         NVIDIA |               FlowNetC | 980 GTX | backward |  75.757 ms |  76.873 ms |
 
 There is still room for optimization, stay tuned on this !

## CPU Benchmark

  * No other implementation is avalaible on CPU.

 | Correlation parameters |               device |     pass |    min time |    avg time |
 | ---------------------- | -------------------- | -------- | ----------: | ----------: |
 |                default | E5-2630 v3 @ 2.40GHz |  forward |  618.303 ms |  626.618 ms |
 |                default | E5-2630 v3 @ 2.40GHz | backward | 1052.563 ms | 1083.407 ms |
 |               FlowNetC | E5-2630 v3 @ 2.40GHz |  forward |  339.769 ms |  354.526 ms |
 |               FlowNetC | E5-2630 v3 @ 2.40GHz | backward |  776.335 ms |  785.781 ms |
