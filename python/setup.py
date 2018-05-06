import os.path
import codecs
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


# Obscure magic required to allow numpy be used as an 'setup_requires'.
class build_ext(_build_ext):
    def finalize_options(self):
        super(build_ext, self).finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# The above obscure magic doesn't seem to work on py2 and prevents the
# extension from building at all, so here's a nasty workaround:
includes = ["lib"]
try:
    import numpy
    includes.append(numpy.get_include())
except ImportError:
    pass

_kastore_module = Extension(
    '_kastore',
    sources=["_kastoremodule.c", "lib/kastore.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=includes,
)

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kastore',
    description='A write-once-read-many store for simple numerical data',
    long_description=long_description,
    url='https://github.com/tskit-dev/kastore',
    author='tskit developers',
    version='0.1.0',
    # TODO setup a tskit developers email address.
    author_email='jerome.kelleher@well.ox.ac.uk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='Binary store numerical arrays',
    packages=['kastore'],
    ext_modules=[_kastore_module],
    install_requires=['six', 'numpy', 'humanize'],
    entry_points={
        'console_scripts': [
            'kastore=kastore.__main__:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/tskit-dev/kastore/issues',
        'Source': 'https://github.com/tskit-dev/kastore',
    },
    setup_requires=['numpy'],
)
