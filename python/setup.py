import codecs
import os.path

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext


# Obscure magic required to allow numpy be used as a 'setup_requires'.
# Based on https://stackoverflow.com/questions/19919905
class local_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        import builtins

        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


_kastore_module = Extension(
    "_kastore",
    sources=["_kastoremodule.c", "lib/kastore.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=["lib"],
)

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

# After exec'ing this file we have kastore_version defined.
kastore_version = None  # Keep PEP8 happy.
version_file = os.path.join("kastore", "_version.py")
with open(version_file) as f:
    exec(f.read())

numpy_ver = "numpy>=1.7"

setup(
    name="kastore",
    description="A write-once-read-many store for simple numerical data",
    long_description=long_description,
    url="https://github.com/tskit-dev/kastore",
    author="tskit developers",
    version=kastore_version,
    # TODO setup a tskit developers email address.
    author_email="jerome.kelleher@well.ox.ac.uk",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="Binary store numerical arrays",
    packages=["kastore"],
    include_package_data=True,
    ext_modules=[_kastore_module],
    install_requires=[numpy_ver, "humanize"],
    entry_points={"console_scripts": ["kastore=kastore.__main__:main"]},
    project_urls={
        "Bug Reports": "https://github.com/tskit-dev/kastore/issues",
        "Source": "https://github.com/tskit-dev/kastore",
    },
    setup_requires=["oldest-supported-numpy"],
    cmdclass={"build_ext": local_build_ext},
    license="MIT",
    platforms=["POSIX", "Windows", "MacOS X"],
)
