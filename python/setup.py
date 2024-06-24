import numpy
from setuptools import Extension
from setuptools import setup

_kastore_module = Extension(
    "_kastore",
    sources=["_kastoremodule.c", "lib/kastore.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=["lib", numpy.get_include()],
)

setup(
    ext_modules=[_kastore_module],
)
