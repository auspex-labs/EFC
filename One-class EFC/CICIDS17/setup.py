import numpy
from Cython.Build import cythonize
from Cython.Compiler.Options import annotate, get_directive_defaults
from Cython.Distutils import build_ext
from setuptools import Extension, setup

directive_defaults = get_directive_defaults()
annotate = True

directive_defaults["linetrace"] = True
directive_defaults["binding"] = True

ext_modules1 = [
    Extension(
        name="classification_functions",
        sources=["classification_functions.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("CYTHON_TRACE", "1")],
    )
]

ext_modules2 = [
    Extension(
        name="dca_functions",
        sources=["dca_functions.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("CYTHON_TRACE", "1")],
    )
]

setup(
    name="Classification functions app",
    ext_modules=cythonize(ext_modules1, annotate=True),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)

setup(
    name="Dca functions app",
    ext_modules=cythonize(ext_modules2, annotate=True),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)
