from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from path import Path
project_root = Path('Correlation_Module')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spatial_correlation_sampler',
    version="0.0.3",
    author="Clément Pinard",
    author_email="clement.pinard@ensta-paristech.fr",
    description="Correlation module for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
    install_requires=['torch>=0.4.1','numpy'],
    ext_modules=[
        CUDAExtension('spatial_correlation_sampler_backend',
                      [project_root/'correlation.cpp',
                       project_root/'correlation_sampler.cpp',
                       project_root/'correlation_cuda_kernel.cu'],
                      extra_compile_args={'cxx': ['-fopenmp'], 'nvcc':[]},
                      extra_link_args=['-lgomp'])
    ],
    packages=[project_root/'spatial_correlation_sampler'],
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
