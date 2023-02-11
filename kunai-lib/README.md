# Kunai Library

This folder represents the library from the Kunai-Static-Analyzer project, Kunai lib is intended to analyze Dex and Apk files. Projects using Kunai can compile this folder and then use the compiled library and the headers.

## Installation

Kunai uses *CMake* to control compilation proces, so you can use this building system for compiling Kunai. The next command is used to configure the compilation of Kunai:

```console
cmake -B build -S .
```

Also we can change the build type, by default this will be *Release*, but *Debug* is recommended if you are testing and developing Kunai:

```console
# for Release compilation
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
# for Debuc compilation
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug
```

Or in the case you also want to compile the *unit test* folder:

```console
cmake -S . -B build -D CMAKE_BUILD_TYPE=DEBUG -D UNIT_TESTING=ON
```

Now we build the libraries and all the other binaries from Kunai, we can specify with *-j* the number of processes to run in parallel so the compilation goes faster:

```console
cmake --build build -j<number of processes>
```

## Project Structure

The project follows a structure on its folders in order to make easier writing new components for the library. We have try to use a similar structure to LLVM project, so you can find the next folders:

* lib/: main code of the application, here you will find the *.cpp* files.
    * Utils/: code from different utility classes.
    * DEX/: code for analysis of DEX files.
        * parser/: parser of DEX format.
        * DVM/: different enums and utilities for managing the Dalvik types.
* include/: all the .hpp and .def files.
    * Kunai/: headers of Kunai library.
        * Utils/: different header of utilities.
        * DEX/: headers for DEX analysis.
            * parser/
            * DVM/
* unit-tests: code for doing unit testing.
* externals: dependencies from Kunai library.

