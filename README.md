# `lagomorph`: Large Scale Computational Anatomy

This project is a revival of ideas implemented in
[PyCA](https://bitbucket.org/scicompanat/pyca) for computational anatomy. In
this repo I plan to reimplement some of these using PyCUDA, and possibly
PyTorch, etc. Importantly, performance will be considered from the start, with
large scale 2D and 3D deformable image registration, atlas building, etc. a main
goal.

# Installation

To install, simply run the following from the top level of this repository:

```
pip install --upgrade .
```

This will pull in the following prerequisites:

- PyCUDA
- numpy

To run the test suite, execute the following command from this directory:

```
python setup.py test
```

# Design

In PyCA, an `Image3D` was the basic type, which is convenient when you have many
images all with different coordinate grids. In `lagomorph`, all operations are
assumed to be happening in batches. As a result, you likely will need to
standardize your images onto a common grid before using. For simplicity,
lagomorph also doesn't support irregular grids. This means the fundamental type
of an image is simply a numpy ndarray or gpuarray of the appropriate dimension
(we ignore positioning in general unless it's needed for an operation).

We do not want to be in the business of implementing optimization routines.
Instead, we will implement for example everything to map vector momenta to a
deformed atlas, and wrap that in a tensorflow or pytorch Tensor type so you can
use your most familiar system to optimize those or combine them with deep neural
networks.

