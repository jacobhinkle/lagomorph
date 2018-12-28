from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lagomorph",
    version="1.0.1",
    author='Jacob Hinkle',
    author_email='jacob.hinkle@gmail.com',
    url='https://github.com/jacobhinkle/lagomorph',
    keywords='medical-imaging computation-anatomy image-registration computer-vision pytorch cuda',
    packages=['lagomorph'],
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
            'lagomorph/cuda_extension/atomic.cuh',
            'lagomorph/cuda_extension/defs.cuh',
            'lagomorph/cuda_extension/diff.cuh',
            'lagomorph/cuda_extension/extrap.cuh',
            'lagomorph/cuda_extension/interp.cuh',
            'lagomorph/cuda_extension/cuda_extension.cpp'
        ],
        include_dirs=["lagomorph/cuda_extension"])]
)
