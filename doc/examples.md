@page examples Examples

[TOC]

@section compiling Compiling the examples

All examples have been compiled on a MacBook Pro with an Apple M2 Max chip and [HDF5](https://www.hdfgroup.org/solutions/hdf5/) 1.14.3
installed via [homebrew](https://brew.sh/).
We further used clang 17.0.6 together with cmake 3.29.1.

Assuming that **nda** has been installed locally (see @ref installation) and that the actual example code is in a file `main.cpp`,
the following generic `CMakeLists.txt` should work for all examples (see also @ref integration):

```cmake
cmake_minimum_required(VERSION 3.20)
project(example CXX)

# set required standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find nda
find_package(nda REQUIRED CONFIG)

# build the example
add_executable(ex main.cpp)
target_link_libraries(ex nda::nda_c)
```
