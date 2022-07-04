import os.path

import numpy
from setuptools import Extension
from setuptools import setup


_kastore_module = Extension(
    "_kastore",
    sources=["_kastoremodule.c", "lib/kastore.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=["lib", numpy.get_include()],
)


# After exec'ing this file we have kastore_version defined.
kastore_version = None  # Keep PEP8 happy.
version_file = os.path.join("kastore", "_version.py")
with open(version_file) as f:
    exec(f.read())

setup(
    # The package name along with all the other metadata is specified in setup.cfg
    # However, GitHub's dependency graph can't see the package unless we put this here.
    name="kastore",
    version=kastore_version,
    ext_modules=[_kastore_module],
)
