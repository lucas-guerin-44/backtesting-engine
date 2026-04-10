"""Build Cython extensions for the backtesting engine.

Usage:
    pip install cython numpy
    python setup.py build_ext --inplace

The Cython extension is optional — the engine falls back to pure Python
if the compiled module is not available.
"""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy as np

    extensions = cythonize(
        [
            Extension("backtesting._core", ["backtesting/_core.pyx"],
                       include_dirs=[np.get_include()]),
            Extension("backtesting._tick_core", ["backtesting/_tick_core.pyx"],
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

setup(
    name="backtesting-engine",
    ext_modules=extensions,
    packages=[],  # Don't auto-discover — we only need the C extension
)
