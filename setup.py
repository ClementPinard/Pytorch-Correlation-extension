import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from os.path import join

CPU_ONLY = False
project_root = 'Correlation_Module'

source_files = ['correlation.cpp', 'correlation_sampler.cpp']

cxx_args = ['-std=c++14', '-fopenmp']

def generate_nvcc_args(gpu_archs):
    nvcc_args = []
    for arch in gpu_archs:
        nvcc_args.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])
    return nvcc_args

gpu_arch = os.environ.get('GPU_ARCH', '').split()
nvcc_args = generate_nvcc_args(gpu_arch)

with open("README.md", "r") as fh:
    long_description = fh.read()

def launch_setup():
    if CPU_ONLY:
        Extension = CppExtension
        macro = []
    else:
        Extension = CUDAExtension
        source_files.append('correlation_cuda_kernel.cu')
        macro = [("USE_CUDA", None)]

    sources = [join(project_root, file) for file in source_files]

    setup(
        name='spatial_correlation_sampler',
        ext_modules=[
            Extension('spatial_correlation_sampler_backend',
                      sources,
                      define_macros=macro,
                      extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
                      extra_link_args=['-lgomp'])
        ],
        cmdclass={'build_ext': BuildExtension},
    )

if __name__ == '__main__':
    launch_setup()
