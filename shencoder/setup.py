import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3',
    '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
else:
    raise NotImplementedError("Unsupported OS")

setup(
    name='shencoder',  # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='_shencoder',  # extension name, import this to use CUDA API
            sources=[
                os.path.join(_src_path, 'src', f) for f in [
                    'shencoder.cu',
                    'bindings.cpp',
                ]
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    })
