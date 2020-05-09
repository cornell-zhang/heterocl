set -e
set -u

make -j${CPU_COUNT} VERBOSE=1
