import argparse
import torch

from torch.autograd import gradcheck
from spatial_correlation_sampler import SpatialCorrelationSamplerFunction

parser = argparse.ArgumentParser()
parser.add_argument('backend', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-k', '--kernel-size', type=int, default=3)
parser.add_argument('--patch', type=int, default=1)
parser.add_argument('--patch_dilation', type=int, default=1)
parser.add_argument('-c', '--channel', type=int, default=10)
parser.add_argument('--height', type=int, default=10)
parser.add_argument('-w', '--width', type=int, default=10)
parser.add_argument('-s', '--stride', type=int, default=1)
parser.add_argument('-p', '--pad', type=int, default=1)

args = parser.parse_args()

input1 = torch.randn(args.batch_size,
                     args.channel,
                     args.height,
                     args.width).double().to(torch.device(args.backend))
input2 = torch.randn(args.batch_size,
                     args.channel,
                     args.height,
                     args.width).double().to(torch.device(args.backend))
input1.requires_grad = True
input2.requires_grad = True

correlation_sampler = SpatialCorrelationSamplerFunction(
    args.kernel_size,
    args.patch,
    args.stride,
    args.pad,
    args.patch_dilation)


if gradcheck(correlation_sampler, [input1, input2]):
    print('Ok')
