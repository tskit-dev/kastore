#!/bin/bash
# Base on https://github.com/pypa/python-manylinux-demo
# Script executed within the docker image to build and test the binary wheels.
# Takes the source distribution tarball as a parameter. 
# NOT to be invoked directly. See the build-wheels.sh script for the intended 
# usage.
set -e -x

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/requirements/development.txt
    "${PYBIN}/pip" wheel /io/$1 -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/kastore*.whl; do
    auditwheel repair "$whl" -w /io/release/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install kastore --no-index -f /io/release/wheelhouse
    (cd /io; "${PYBIN}/python" -m nose tests/test_storage.py)
done

# Change the permissions of the wheels from root to the local user.
chown `stat -c "%u:%g" /io` /io/release/wheelhouse/*.whl
