#!/usr/bin/env bash

# make file in bash made mainly for debian
# this should be modified once more OS are
# included.


# taken from capstone
MAKE_JOBS=$((${MAKE_JOBS}+0))
[ ${MAKE_JOBS} -lt 1 ] && \
  MAKE_JOBS=4

COMPILER=g++

check_and_build_dependencies() {
    echo "[+] Checking for package libspdlog-dev"
    dpkg -s libspdlog-dev 2> /dev/null > /dev/null
    if [ $? -ne 0 ]; then
        echo "[-] libspdlog-dev not installed, installing it..."
        sudo apt install libspdlog-dev
    fi
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
    "" ) 
        build_kunai;;
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
        echo "Usage: $0 ["`grep '^  "' $0 | cut -d '"' -f 2 | tr "\\n" "|"`"]"
        exit 1;;
esac