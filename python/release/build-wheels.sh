#!/bin/bash
# Run from the python/ directory with path to the distribution tarball like so:
# ./release/build-wheels.sh dist/kastore-X.Y.Z.tar.gz
# After building, the wheels will be in release/wheelhouse which can then 
# be uploaded to PyPI using twine.
set -e 

if [ -z "$1" ]; then
    echo Usage: ./release/build-wheels.sh TARBALL
    exit 1
fi

docker run -it -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /io/release/docker-build-wheels.sh $1
