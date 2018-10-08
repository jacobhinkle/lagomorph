from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lagomorph",
    version="0.1.4",
    packages=['lagomorph', 'lagomorph.torch'],
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    install_requires=['torch>=0.4.0','pycuda==2017.1.1','numpy','scikit-cuda'],
    tests_require=['pytest'],
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[CUDAExtension('lagomorph_torch_cuda', [
        'lagomorph/torch/affine_cuda.cpp',
        'lagomorph/torch/affine_cuda_kernels.cu'
            ], include_dirs=["lagomorph/torch"])]
)
