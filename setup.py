from setuptools import setup
setup(
    name="lagomorph",
    version="0.1.2",
    packages=['lagomorph'],
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    install_requires=['pycuda','numpy','scikit-cuda'],
    tests_require=['pytest']
)
