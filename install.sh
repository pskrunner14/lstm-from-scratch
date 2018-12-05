#! /bin/bash

# remove build dir if present
rm -rf build
mkdir build

cd build
# start build process and create Makefile
cmake ..
# build project using Makefile
make