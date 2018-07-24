from setuptools import setup
setup(
    name="lagomorph",
    version="0.1",
    packages=['lagomorph'],
    setup_requires=['pytest-runner'],
    install_requires=['pycuda','numpy','scikit-cuda'],
    tests_require=['pytest']
)
