from setuptools import setup
setup(
    name="lagomorph",
    version="0.1.3",
    packages=['lagomorph'],
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    install_requires=['pycuda==2017.1.1','numpy','scikit-cuda'],
    tests_require=['pytest']
)
