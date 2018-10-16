from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# determine compute capability for present gpu
from torch.cuda import get_device_capability
cuda_cap = get_device_capability(0)

setup(
    name="lagomorph",
    version="0.1.7",
    packages=['lagomorph', 'lagomorph.torch'],
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    install_requires=['torch>=0.4.0','pycuda==2017.1.1','numpy','scikit-cuda'],
    tests_require=['pytest'],
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[CUDAExtension('lagomorph_torch_cuda', [
            'lagomorph/torch/affine_cuda_kernels.cu',
            'lagomorph/torch/metric_cuda_kernels.cu',
            'lagomorph/torch/extension.cpp'
        ],
        include_dirs=["lagomorph/torch"],
        extra_compile_args={'cxx': ['-O3'],
            'nvcc': [f'-arch=sm_{cuda_cap[0]}{cuda_cap[1]}']})]
        #extra_cuda_flags=['-use_fast_math'])]
)
