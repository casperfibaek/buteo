from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# python compile_cython.py build_ext --inplace
# cython -a filters.pyx

if os.name == 'nt':  # Windows
    setup(ext_modules=cythonize(cythonize((Extension(
                                "stats_local",
                                sources=["stats_local.pyx"],
                                extra_compile_args=["/O2", "/fp:fast", "/openmp"],
                                extra_link_args=['/openmp']
    )))))

    setup(ext_modules=cythonize(cythonize((Extension(
                                "stats_global",
                                sources=["stats_global.pyx"],
                                extra_compile_args=["/O2", "/fp:fast", "/openmp"],
                                extra_link_args=['/openmp']
    )))))

    setup(ext_modules=cythonize(cythonize((Extension(
                                "stats_local_no_kernel",
                                sources=["stats_local_no_kernel.pyx"],
                                extra_compile_args=["/O2", "/fp:fast", "/openmp"],
                                extra_link_args=['/openmp']
    )))))

else:  # Unix
    setup(ext_modules=cythonize(cythonize((Extension(
                                "stats_local",
                                sources=["stats_local.pyx"],
                                extra_compile_args=["-Wall", "-O3", "-fopenmp", "-Ofast"],
                                extra_link_args=['-fopenmp'],
    )))))


    setup(ext_modules=cythonize(cythonize((Extension(
                                "stats_global",
                                sources=["stats_global.pyx"],
                                extra_compile_args=["-Wall", "-O3", "-fopenmp"],
                                extra_link_args=['-fopenmp'],
    )))))


    setup(ext_modules=cythonize(cythonize((Extension(
                                "stats_local_no_kernel",
                                sources=["stats_local_no_kernel.pyx"],
                                extra_compile_args=["-Wall", "-O3", "-fopenmp"],
                                extra_link_args=['-fopenmp'],
    )))))