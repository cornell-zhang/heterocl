set -e
set -u

make build-tvm -j${CPU_COUNT} VERBOSE=1

cd python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cd ../hlib/python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cd ../../
