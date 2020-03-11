from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
# import numpy

# python setup.py build_ext --inplace
# cython -a filters.pyx

ext_modules = cythonize((Extension(
                            "stats_global",
                            sources=["stats_global.pyx"],
                            # include_dirs=[numpy.get_include()],
                            extra_compile_args=["-Wall", "-O3", "-fopenmp"],
                            extra_link_args=['-fopenmp'],
                        )))

setup(ext_modules=cythonize(ext_modules, annotate=True))

ext_modules = cythonize((Extension(
                            "stats_local",
                            sources=["stats_local.pyx"],
                            # include_dirs=[numpy.get_include()],
                            extra_compile_args=["-Wall", "-O3", "-fopenmp"],
                            extra_link_args=['-fopenmp'],
                        )))

setup(ext_modules=cythonize(ext_modules, annotate=True))
