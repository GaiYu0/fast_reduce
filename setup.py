from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='min2d_2nd_dim',
    ext_modules=[
        CUDAExtension('min2d_2nd_dim_cuda', [
            'min2d_2nd_dim_cuda.cpp',
            'min2d_2nd_dim_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
