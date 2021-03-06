from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from glob import glob
import os
import platform

system = platform.system().lower()

# python compile_cython.py build_ext --inplace
# cython -a filters.pyx

cython_files = glob("./stats/*.pyx")

for f in cython_files:
    basename = os.path.basename(f)
    name = os.path.splitext(basename)[0]

    if system == 'windows':  # Windows
        setup(ext_modules=cythonize(cythonize((Extension(
            name,
            sources=[f],
            extra_compile_args=["/O2", "/fp:fast", "/openmp"],
            extra_link_args=['/openmp']
        )))))
    elif system == 'linux':
        setup(ext_modules=cythonize(cythonize((Extension(
            name,
            sources=[f],
            extra_compile_args=["-Wall", "-O3", "-fopenmp", "-Ofast"],
            extra_link_args=['-fopenmp'],
        )))))
    else:
        setup(ext_modules=cythonize(cythonize((Extension(
            name,
            sources=[f],
        )))))