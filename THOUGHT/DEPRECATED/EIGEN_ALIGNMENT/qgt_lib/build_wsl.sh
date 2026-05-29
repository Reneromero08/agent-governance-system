#!/bin/bash
set -e
SRC="/mnt/d/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/EIGEN_ALIGNMENT/qgt_lib"
BUILD="/tmp/qgt_build"
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"
cmake "$SRC" -DCMAKE_BUILD_TYPE=Release
make -j4 2>&1 | tail -10
echo "=== Library files ==="
find "$BUILD/lib" -name "*.so*" -type f
echo "=== Copying to source build dir ==="
mkdir -p "$SRC/build/lib"
cp "$BUILD"/lib/libquantum_geometric.so* "$SRC/build/lib/"
echo "DONE"
