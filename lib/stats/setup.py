from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# python setup.py build_ext --inplace
# cython -a filters.pyx

ext_modules = cythonize((Extension(
                            "base_filters",
                            sources=["base_filters.pyx"],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=["-Wall", "-O3", "-pthread"],
                        )))

setup(ext_modules=cythonize(ext_modules, annotate=False))

ext_modules = cythonize((Extension(
                            "stats",
                            sources=["stats.pyx"],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=["-Wall", "-O3", "-pthread"],
                        )))

setup(ext_modules=cythonize(ext_modules, annotate=False))