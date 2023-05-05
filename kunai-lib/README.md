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

## MjolnIR

MjolnIR is the Intermediate Representation used in Kunai. In the previous version MjolnIR was written from scratch and was compiled together with Kunai as part of the library. For the refactoring of Kunai, we have also decided changing the design and the development of MjolnIR.

Currently for the design of MjolnIR we have decided creating a *Dialect* using the MLIR Project [1], we can obtain the next definition from its website:

*The MLIR project is a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.*

MLIR allows us defining new types and operations for the IR, and use the infrastructure behind for applying transformations and optimizations. The IRs created with MLIR are known as *Dialects*, the IRs designed with MLIR are in the *Static Single-Assignment form* (SSA), then each one of the values returned by the operations will be assigned only once (in opposite to the Dalvik Bytecode).

MjolnIR dialect contains types that reassemble the types of Java (e.g. *int, double, float, etc*), as well as an opaque *object* type that as one of its attributes contains the name of its class. The operations of MjolnIR are a reduced set of instructions that will try to cover all the different instructions found on th Dalvik instruction set (for the moment not all the instructions are supported).

This part of Kunai is very experimental for the moment and for compiling it you will need to have MLIR on your system, for its compilation and installation you can follow their guidelines in their website [2], but you can use the next command for download and compile MLIR:

```console
$ git clone https://github.com/llvm/llvm-project.git
$ mkdir llvm-project/build
$ cd llvm-project/build
$ cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
   # I highly recommend using CLANG and LLD
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
   # CCache can drastically speed up further rebuilds, try adding:
   -DLLVM_CCACHE_BUILD=ON
   # Please for the moment do not use ASAN as Kunai is not compiled with it
   # and it generated many crashes...
$ cmake --build . --target check-mlir
# finally install it
$ sudo cmake --install .
```

Once we have it installed in our system, I will use the same compiler and the same linker for compiling Kunai with support for MjolnIR, so we will also activate MjolnIR and the unit tests:

```console
$ cd kunai-lib/
# configure the project
$ $ cmake -S . -B build/ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DUSE_MJOLNIR=ON -DCMAKE_CXX_FLAGS="-fuse-ld=lld" -DUNIT_TESTING=ON
-- The CXX compiler identification is Clang 17.0.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/local/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- C++ compiler ID: Clang
-- C++ compiler version: 17.0.0
-- Build Type: Debug
...
# compile it
$ cmake --build build/ -j
```

With this you should have been able to compile Kunai together with MjolnIR, and also the unit tests. As previously stated this part of the project is right now very experimental, and no code warranty is provided, soon we will try to improve the installation of Kunai with MjolnIR and also improve the Lifter for making its usage easier for analysis.


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
* MjolnIR/: Folder with the code of MjolnIR Intermediate Representation, this IR is based on MLIR from LLVM Project([1]).
    * include/: .hpp and .td files from the Intermediate Representation.
        * include/Dalvik/: headers and tablegen files of the MLIR Dialect for the IR.
        * include/Lifter/: headers from Lifter from Dalvik to MjolnIR.
    * lib/: source code from MjolnIR Dialect and Lifter.
        * lib/Dalvik/: source code from MjolnIR Dialect.
        * lib/Lifter/: source code from Lifter from Dalvik to MjolnIR.
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

## Possible issues

Kunai depends on C++-20, so maybe you will need to install newer C++ libraries:

```console
sudo apt install libstdc++-10-dev
```

Also we recommend you using clang++ as compiler (version 16 and newer version 17 tested).

## Authors of Kunai

* Eduardo Blázquez <kunai(dot)static(dot)analysis(at)gmail(dot)com>


[1]: https://mlir.llvm.org/
[2]: https://mlir.llvm.org/getting_started/