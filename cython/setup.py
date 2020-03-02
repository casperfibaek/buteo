from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# python setup.py build_ext --inplace

ext_modules = cythonize((Extension(
                            "c_filter",
                            sources=["c_filter.pyx"],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=["/O2", "/fp:fast", "/openmp"],
                            extra_link_args=['/openmp']
                        )))

setup(
    ext_modules=cythonize(ext_modules,
                          compiler_directives={'language_level': "3"},
                          annotate=True,
                          ),
)
