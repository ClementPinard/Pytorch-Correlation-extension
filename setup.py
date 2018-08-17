from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from path import Path
project_root = Path('Correlation_Module')

setup(
    name='spatial_correlation_sampler',
    version="0.0.1",
    author="Clément Pinard",
    author_email="clement.pinard@ensta-paristech.fr",
    url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
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
    })
