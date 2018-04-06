# setup.py based on https://github.com/pypa/sampleproject
import setuptools
import os.path
import codecs

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def get_numpy_includes():
    import numpy as np
    return np.get_include()


_kastore_module = setuptools.Extension(
    '_kastore',
    sources=["_kastoremodule.c", "lib/kastore.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=[get_numpy_includes(), "lib"],
)


setuptools.setup(
    name='kastore',
    version='0.0.1',
    description='A write-once-read-many store for simple numerical data',
    long_description=long_description,
    url='https://github.com/tskit-dev/kastore',
    author='tskit developers',
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
    install_requires=['numpy'],
    entry_points={
        'console_scripts': [
            'kastore=kastore:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/tskit-dev/kastore/issues',
        'Source': 'https://github.com/tskit-dev/kastore',
    },
)
