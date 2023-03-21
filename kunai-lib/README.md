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

Or in the case you also want to compile the *unit-tests* folder:

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
        * analysis/: sources of the different analysis for DEX format.
* include/: all the .hpp and .def files.
    * Kunai/: headers of Kunai library.
        * Utils/: different header of utilities.
        * DEX/: headers for DEX analysis.
            * parser/
            * DVM/
            * analysis/
* unit-tests: code for doing unit testing.
* externals: dependencies from Kunai library.

## Library Paper

A paper from the project was published in SoftwareX, you can find the paper in the next [link](https://www.sciencedirect.com/science/article/pii/S2352711023000663). While the presented library is an old version, it is possible to follow the idea behind Kunai's lib. You can reference the paper with the next bibtex:

```
@article{BLAZQUEZ2023101370,
title = {Kunai: A static analysis framework for Android apps},
journal = {SoftwareX},
volume = {22},
pages = {101370},
year = {2023},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2023.101370},
url = {https://www.sciencedirect.com/science/article/pii/S2352711023000663},
author = {Eduardo Blázquez and Juan Tapiador},
keywords = {Android, Static analysis, Software analysis, Mobile apps},
abstract = {This paper describes Kunai, a C++ library that offers static analysis functionalities for Android apps. Kunai’s main goal is to provide analysts and researchers with a framework to conduct advanced static analysis of Dalvik code, including parsing, disassembling and code analysis. Written in C++, it focuses on efficiency and an extensible software architecture that enables the integration of new analysis modules easily. Kunai is particularly suitable for the development of analysis pipelines that need to scale up to process large app datasets.}
}
```