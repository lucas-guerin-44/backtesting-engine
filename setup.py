"""Cython extension build hook.

Package metadata, dependencies, and package discovery live in
``pyproject.toml``. This file exists only to compile the optional
``backtesting._core`` extension.

Usage:
    pip install cython numpy
    python setup.py build_ext --inplace

The Cython extension is optional — the engine falls back to pure Python
if the compiled module is not available. ``_core`` accelerates the canonical
indicator kernels (EMA/ATR/RSI) and the vectorized trade-chaining loop.
"""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy as np

    extensions = cythonize(
        [
            Extension("backtesting._core", ["backtesting/_core.pyx"],
                       include_dirs=[np.get_include()]),
        ],
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except ImportError:
    extensions = []
    print("Cython not found — skipping C extension build.")

setup(ext_modules=extensions)
