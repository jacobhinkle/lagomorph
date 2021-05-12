# lagomorph: Large Scale Computational Anatomy in PyTorch

[![PyPI version](https://badge.fury.io/py/lagomorph.svg)](https://badge.fury.io/py/lagomorph)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Lagomorph aims to provide tools for [computational
anatomy](https://en.wikipedia.org/wiki/Computational_anatomy) in the context of
the deep learning framework PyTorch, which will enable easier integration of
image registration methodologies with deep learning.

This project follows past LDDMM implementations including CompOnc, AtlasWerks, and [PyCA](https://bitbucket.org/scicompanat/pyca). Lagomorph makes those methods more amenable to use with [PyTorch](https://pytorch.org).
Our goal is to enable large-scale 2D
and 3D deformable image registration, atlas building, etc. to be used seamlessly with deep neural networks implemented in PyTorch. As such, we follow PyTorch conventions, storing tensors in NCDHW format, and using built-in PyTorch functionality where possible.

**NOTE:** Currently, only computation on CUDA GPUs is supported, but CPU and AMD GPU support is planned.

# Installation

To install, simply run the following:

```
pip install lagomorph
```

This will pull in the following prerequisites:

- PyTorch >= 0.4.0
- numpy

Note that often it is necessary to install `numpy` manually first using `pip
install numpy` due to weirdness in numpy's packaging.

To run the test suite for lagomorph, execute the following command from the
current directory in this repository:

```
python setup.py test
```

# Design Notes

In PyCA, an `Image3D` was the basic type, which is convenient when you have many
images all with different coordinate grids. In `lagomorph`, all operations are
assumed to be happening in batches. As a result, you likely will need to
standardize your images onto a common grid before using. For simplicity, at this
time lagomorph also doesn't support irregular grids. This means the fundamental
type of an image is simply a `torch.Tensor` of the appropriate dimension in
NCWH(D) order, where a scalar image is simply a single channel image and a
vector field is a `dim`-channel image. We ignore positioning with origin and
voxel spacing in general unless it's needed for an operation.

# Related projects

Lagomorph was heavily influenced by the [PyCA](https://bitbucket) project.
