from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spatial_correlation_sampler',
    ext_modules=[
        CUDAExtension('spatial_correlation_sampler_backend',
                      ['correlation.cpp',
                       'correlation_sampler.cpp',
                       'correlation_cuda_kernel.cu'],
                      extra_compile_args={'cxx': ['-fopenmp'], 'nvcc':[]},
                      extra_link_args=['-lgomp'])
    ],
    packages=['spatial_correlation_sampler'],
    cmdclass={
        'build_ext': BuildExtension
    })
