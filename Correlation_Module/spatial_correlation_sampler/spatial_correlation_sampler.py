import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import spatial_correlation_sampler_backend as correlation


def spatial_correlation_sample(input1,
                               input2,
                               kernel_size=1,
                               patch_size=1,
                               stride=1,
                               padding=0,
                               dilation_patch=1):
    corr_func = SpatialCorrelationSamplerFunction(kernel_size,
                                                  patch_size,
                                                  stride,
                                                  padding,
                                                  dilation_patch)
    return corr_func(input1, input2)


class SpatialCorrelationSamplerFunction(Function):
    def __init__(self,
                 kernel_size,
                 patch_size,
                 stride,
                 padding,
                 dilation_patch):
        super(SpatialCorrelationSamplerFunction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation_patch = _pair(dilation_patch)

    def forward(self, input1, input2):

        self.save_for_backward(input1, input2)
        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride

        output = correlation.forward(input1, input2,
                                     kH, kW, patchH, patchW,
                                     padH, padW, dilation_patchH, dilation_patchW,
                                     dH, dW)

        return output

    @once_differentiable
    def backward(self, grad_output):
        input1, input2 = self.saved_variables

        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride

        grad_input1, grad_input2 = correlation.backward(input1, input2, grad_output,
                                                        kH, kW, patchH, patchW,
                                                        padH, padW,
                                                        dilation_patchH, dilation_patchW,
                                                        dH, dW)
        return grad_input1, grad_input2


class SpatialCorrelationSampler(nn.Module):
    def __init__(self, kernel_size=1, patch_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(SpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return spatial_correlation_sample(input1, input2, self.kernel_size,
                                          self.patch_size, self.stride,
                                          self.padding, self.dilation_patch)
