from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lagomorph",
    author='Jacob Hinkle',
    author_email='hinklejd@ornl.gov',
    url='https://github.com/jacobhinkle/lagomorph',
    keywords='medical-imaging computation-anatomy image-registration computer-vision pytorch cuda',
    packages=['lagomorph'],
    python_requires=">=3.6",
    use_scm_version=True,
    setup_requires=['pytest-runner','setuptools_scm'],
    install_requires=['torch>=0.4.0','numpy'],
    tests_require=['pytest'],
    cmdclass={'build_ext': BuildExtension},
    entry_points={'console_scripts':['lagomorph=lagomorph.__main__:LagomorphTool']},
    ext_modules=[CUDAExtension('lagomorph_ext', [
            'lagomorph/extension/cpu/affine.cpp',
            'lagomorph/extension/cuda/affine.cu',
            'lagomorph/extension/cuda/diff.cu',
            'lagomorph/extension/cuda/interp.cu',
            'lagomorph/extension/cuda/metric.cu',
            'lagomorph/extension/extension.cpp'
        ],
        include_dirs=["lagomorph/extension/include"])]
)
