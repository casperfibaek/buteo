from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# python setup.py build_ext --inplace
# cython -a filters.pyx

ext_modules = cythonize((Extension(
                            "filters",
                            sources=["./lib/cython/filters.pyx"],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=["/O2", "/fp:fast", "/openmp"],
                        )))

setup(ext_modules=cythonize(ext_modules, annotate=True))
