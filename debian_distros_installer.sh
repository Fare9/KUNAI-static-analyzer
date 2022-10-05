#!/usr/bin/env bash

# Installer for debian based systems
# this can include systems like:
# Ubuntu, Elementary OS...


# taken from capstone
MAKE_JOBS=$((${MAKE_JOBS}+0))
[ ${MAKE_JOBS} -lt 1 ] && \
  MAKE_JOBS=4

LIB_ZIP=libzip.so
LIB_SPDLOG=libspdlog.a

STATIC_LIB_NAME=libkunai.a
SHARED_LIB_NAME=libkunai.so

BIN_FOLDER=bin/
INCLUDE_FOLDER=src/include/KUNAI/

# by default g++, but can be changed by
# clang++
COMPILER=g++

# check if the dependencies exist
# if not install them, try to compile
# if apt installation is not possible.
check_and_build_dependencies()
{
    echo "[!] DEBIAN/Ubuntu/Elementary OS dependecy installation"
    
    echo "[+] Installing CMAKE"
    sudo apt install cmake -y

    echo "[+] Checking for package libspdlog-dev"
    dpkg -s libspdlog-dev 2> /dev/null > /dev/null

    if [ $? -ne 0 ]; then
        echo "[-] libspdlog-dev not installed, trying to install by apt version 1:1.5.0"
        sudo apt install libspdlog-dev=1:1.5.0-1 -y
    fi # end of libspdlog with apt

    if [ $? -ne 0 ]; then
        echo "[-] There was a problem installing by apt, installing from sources"

        if [ ! -f external/spdlog/ ]; then
            echo "[-] Not found spdlog folder, cloning from repo"
            git clone https://github.com/gabime/spdlog.git ./external/spdlog/ --branch v1.5.0 --single-branch
        fi

        if [ ! -f external/spdlog/build/${LIB_SPDLOG} ]; then
            echo "[-] Not found compiled library, compiling"
            current_dir=${PWD}
            echo "[+] Moving to external/spdlog/"
            cd external/spdlog/
            echo "[+] Creating new build folder"
            mkdir build
            cd build
            echo "[+] Compiling library"
            cmake ..
            make -j ${MAKE_JOBS}
            make install
            echo "[+] Returning to ${current_dir}"
            cd ${current_dir}
        fi
    fi # end of libspdlog from source

    if [ ! -f external/zip/ ]; then
        echo "[-] Not found zip folder, cloning from repo..."
        git clone https://github.com/kuba--/zip.git ./external/zip/
    fi

    if [ ! -f external/zip/build/${LIB_ZIP} ]; then
        echo "[-] Not found compiled library, compiling"
        current_dir=${PWD}
        echo "[+] Moving to external/zip/"
        cd external/zip/
        echo "[+] Creating new build folder"
        mkdir build
        cd build
        echo "[+] Compiling library"
        cmake -DBUILD_SHARED_LIBS=true ..
        cmake --build .
        echo "[+] Returning to ${current_dir}"
        cd ${current_dir}
    fi

    echo "[+] Checking for ${LIB_ZIP}"
    if [ ! -f /usr/lib/${LIB_ZIP} ]; then
        echo "[-] ${LIB_ZIP} not found, installing it"
        sudo cp external/zip/build/${LIB_ZIP} /usr/lib/${LIB_ZIP}
    fi

    echo "[!] Finished installing dependencies"
}

# build Kunai in debug mode
build_debug() 
{
    # check if dependencies are not installed.
    check_and_build_dependencies

    echo "[+] Compiling KUNAI with ${MAKE_JOBS} jobs"

    make clean

    CXX=${COMPILER} make debug --silent -j${MAKE_JOBS}

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred, check output..."
        exit 1
    fi

    echo "[+] KUNAI compiled correctly"
}

# build kunai in "release" mode
build_kunai() 
{
    # check if dependencies are not installed.
    check_and_build_dependencies

    echo "[+] Compiling KUNAI with ${MAKE_JOBS} jobs"

    make clean

    CXX=${COMPILER} make --silent -j${MAKE_JOBS}

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred, check output..."
        exit 1
    fi

    echo "[+] KUNAI compiled correctly"
}

uninstall_kunai()
{
    echo "[+] Uninstalling KUNAI from system"
    
    if [ -f /usr/include/KUNAI ]; then
        sudo rm -rf /usr/include/KUNAI
    fi

    if [ -f /usr/lib/${SHARED_LIB_NAME} ]; then
        sudo rm -f /usr/lib/${SHARED_LIB_NAME}
    fi

    if [ -f /usr/lib/${STATIC_LIB_NAME} ]; then
        sudo rm -f /usr/lib/${STATIC_LIB_NAME}
    fi
}

# install kunai on system
install_kunai() 
{
    echo "[+] Installing KUNAI, compiling first..."
    build_kunai
    echo "[+] Installing on system"
    echo "[+] Removing previous folders"
    uninstall_kunai
    echo "[+] Copying libs to /usr/lib"
    sudo cp ${BIN_FOLDER}${STATIC_LIB_NAME} /usr/lib
    sudo cp ${BIN_FOLDER}${SHARED_LIB_NAME} /usr/lib
    sudo cp -r ${INCLUDE_FOLDER} /usr/include/

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred while installing..."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred while installing..."
        exit 1
    fi
}

if [ ! -f "/etc/debian_version" ]; then
    echo "[-] This is not the script your system is looking for..."
    exit 1
fi

TARGET="$1"
[ -n "$TARGET" ] && shift

case "$TARGET" in
    "build" ) 
        build_kunai;;
    "debug" ) 
        build_debug;;
    "install" ) 
        install_kunai;;
    "uninstall" )
        uninstall_kunai;;
    "dependencies" ) 
        check_and_build_dependencies;;
    "clang" ) 
        COMPILER=clang++ 
        build_kunai;;
    "clang_debug" )
        COMPILER=clang++
        build_debug;;
    "run_tests" )
        run_tests;;
    * )
        echo "Usage: $0 <option>"
        echo "Next options are available:"
        echo "    - build: build kunai directly running Makefile."
        echo "    - debug: build kunai with symbols and verbose debug messages directly running Makefile."
        echo "    - install: build kunai and install it on system."
        echo "    - uninstall: remove kunai from system."
        echo "    - dependencies: check if dependencies are present, if not, install them."
        echo "    - clang: build kunai but this time using clang++ as compiler."
        echo "    - clang_debug: build kunai in debug mode but this time using clang++ as compiler."
        echo "    - run_tests: run the 'run_tests.sh' script."

        exit 1;;
esac