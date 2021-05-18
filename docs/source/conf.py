# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from setuptools_scm import get_version

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# added to find main lagomorph modules
sys.path.insert(0, os.path.abspath("../.."))
# this holds a dummy extension file that we use to avoid building on RTFD
sys.path.insert(0, os.path.abspath("sphinxpypath"))


# -- Project information -----------------------------------------------------

project = "lagomorph"
copyright = "2021, Jacob Hinkle"
author = "Jacob Hinkle"

# The full version, including alpha/beta/rc tags
release = get_version(root="../..", relative_to=__file__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_math_dollar",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

mathjax3_config = {
    "tex2jax": {"inlineMath": [["\\(", "\\)"]], "displayMath": [["\\[", "\\]"]]},
    "tex": {
        "macros": {
            "Ad": r"\operatorname{Ad}",
            "ad": r"\operatorname{ad}",
            "Diff": r"\operatorname{Diff}",
            "div": r"\operatorname{div}",
            "R": r"\mathbb{R}",
            "sym": r"\operatorname{sym}",
        }
    },
}

primary_domain = "py"
highlight_language = "python"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

nitpick_ignore = [
    ("py:class", "builtins.bool"),
    ("py:class", "builtins.float"),
    ("py:class", "builtins.int"),
    ("py:class", "builtins.str"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "builtins": ("https://docs.python.org/3", None),
}

master_doc = "index"
