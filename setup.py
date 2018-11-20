from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# determine compute capability for present gpu
from torch.cuda import get_device_capability
cuda_cap = get_device_capability(0)

setup(
    name="lagomorph",
    version="0.1.16",
    packages=['lagomorph'],
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    install_requires=['torch>=0.4.0','numpy'],
    tests_require=['pytest'],
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[CUDAExtension('lagomorph_cuda', [
            'lagomorph/cuda_extension/affine_cuda_kernels.cu',
            'lagomorph/cuda_extension/diff_cuda_kernels.cu',
            'lagomorph/cuda_extension/interp_cuda_kernels.cu',
            'lagomorph/cuda_extension/metric_cuda_kernels.cu',
            'lagomorph/cuda_extension/cuda_extension.cpp'
        ],
        include_dirs=["lagomorph/cuda_extension"],
        extra_compile_args={'cxx': ['-O3'],
            'nvcc': [f'-arch=sm_{cuda_cap[0]}{cuda_cap[1]}']})]
        #extra_cuda_flags=['-use_fast_math'])]
)
