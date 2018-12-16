# lagomorph: Large Scale Computational Anatomy in PyTorch

[![PyPI version](https://badge.fury.io/py/lagomorph.svg)](https://badge.fury.io/py/lagomorph)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Lagomorph aims to provide tools for [computational
anatomy](https://en.wikipedia.org/wiki/Computational_anatomy) in the context of
the deep learning framework PyTorch, which will enable easier integration of
image registration methodologies with deep learning.

This project is a revival of ideas implemented through the years in CompOnc,
then AtlasWerks, then in [PyCA](https://bitbucket.org/scicompanat/pyca). In this
repo I have reimplemented some of these using [PyTorch](https://pytorch.org).
Importantly, performance will be considered from the start, with large scale 2D
and 3D deformable image registration, atlas building, etc. a main goal.

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

# History

Some time around 2010, a collection of graduate students at the [SCI
Institute](https://sci.utah.edu) involved in computational anatomy research
decided to rewrite a lot of C++/CUDA code in a way that could be used from
Python. This led to us writing a new C++ library that came with a SWIG wrapper
for python, which greatly increased our productivity. We named it
[PyCA](https://bitbucket), which officially stands for Python for Computational
Anatomy. But really, the name was always an homage to Sam Preston's wonderful
feline companion Pika, who himself was named after a species of lovable rodents
we encountered often in the mountains around Salt Lake City on our many trips
together.

PyCA was a useful project but in 2018 is long in the tooth. After some quick
attempts to update it to be compatible with Python 3, I gave up and decided to
try integrating with PyCUDA. That experiment was successful, but due to GPU
context issues, that version was a headache to wrangle into a form that PyTorch
or tensorflow would play nicely with (though that old PyCUDA code still exists
in this repository in versions prior to v0.1.15). Instead, I decided to embrace
PyTorch, whose ATen C++ library makes it very easy to support multiple backends
(we plan to provide an AMD ROCm version at some point). This allows us to
leverage all the engineering done by PyTorch contributors toward NCCL2 support
and other challenges.
