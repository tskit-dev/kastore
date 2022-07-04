#!/bin/bash
DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DOCKER_DIR/shared.env"

set -e -x

ARCH=`uname -p`
echo "arch=$ARCH"
cd python

for V in "${PYTHON_VERSIONS[@]}"; do
    PYBIN=/opt/python/$V/bin
    rm -rf build/       # Avoid lib build by narrow Python is used by wide python
    $PYBIN/python -m venv env
    source env/bin/activate
    $PYBIN/python -m pip install --upgrade build
    SETUPTOOLS_SCM_DEBUG=1 $PYBIN/python -m build
done

cd dist
for whl in *.whl; do
    auditwheel repair "$whl"
    rm "$whl"
done