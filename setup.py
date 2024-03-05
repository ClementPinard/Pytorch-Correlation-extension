import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from os.path import join

CPU_ONLY = False
project_root = 'Correlation_Module'

source_files = ['correlation.cpp', 'correlation_sampler.cpp']

cxx_args = ['-std=c++17', '-fopenmp']

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
        version="0.5.0",
        author="Clément Pinard",
        author_email="mail@clementpinard.fr",
        description="Correlation module for pytorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
        install_requires=['torch>=1.1', 'numpy'],
        ext_modules=[
            Extension('spatial_correlation_sampler_backend',
                      sources,
                      define_macros=macro,
                      extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
                      extra_link_args=['-lgomp'])
        ],
        package_dir={'': project_root},
        packages=['spatial_correlation_sampler'],
        cmdclass={
            'build_ext': BuildExtension
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ])


if __name__ == '__main__':
    launch_setup()
