#!/usr/bin/env bash

# make file in bash made mainly for debian
# this should be modified once more OS are
# included.


# taken from capstone
MAKE_JOBS=$((${MAKE_JOBS}+0))
[ ${MAKE_JOBS} -lt 1 ] && \
  MAKE_JOBS=4

COMPILER=g++

LIB_ZIP=libzip.so


check_and_build_dependencies() {
    echo "[+] Checking for package libspdlog-dev"
    dpkg -s libspdlog-dev 2> /dev/null > /dev/null
    if [ $? -ne 0 ]; then
        echo "[-] libspdlog-dev not installed, installing it..."
        sudo apt install libspdlog-dev=1:1.5.0-1
    fi

    if [ ! -f external/zip/ ]; then
        echo "[-] Not found zip folder, cloning from repo..."
        git clone https://github.com/kuba--/zip.git ./external/zip/
    fi

    sudo apt install cmake

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


build_debug() {
    # check if dependencies are not installed.
    check_and_build_dependencies

    echo "[+] Compiling KUNAI with ${MAKE_JOBS} jobs"

    CXX=${COMPILER} make debug --silent -j${MAKE_JOBS}

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred, check output..."
        exit 1
    fi

    echo "[+] KUNAI compiled correctly"
}


build_kunai() {
    # check if dependencies are not installed.
    check_and_build_dependencies

    echo "[+] Compiling KUNAI with ${MAKE_JOBS} jobs"

    CXX=${COMPILER} make --silent -j${MAKE_JOBS}

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred, check output..."
        exit 1
    fi

    echo "[+] KUNAI compiled correctly"
}

install_kunai() {
    # install kunai on system

    echo "[+] Installing KUNAI, compiling first..."
    build_kunai
    echo "[+] Installing on system"
    make install

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred while installing..."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred while installing..."
        exit 1
    fi
}

uninstall_kunai() {
    echo "[+] Uninstalling KUNAI from system"
    make remove

    if [ $? -ne 0 ]; then
        echo "[-] An error ocurred while uninstalling..."
        exit 1
    fi

    echo "[+] KUNAI uninstalled correctly"
}

run_tests() {
    echo "[+] Running tests"
    ./run_tests.sh
}

[ -z "${UNAME}" ] && UNAME=$(uname)

TARGET="$1"
[ -n "$TARGET" ] && shift


if [ ${UNAME} != "Linux" ]; then
    echo "[-] System not supported yet"
    exit 1
fi

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
        echo "    - run_tests: run the 'run_tests.sh' script."

        exit 1;;
esac