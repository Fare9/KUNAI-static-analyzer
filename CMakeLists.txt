cmake_minimum_required(VERSION 3.15)

# Project name for KUNAI
project(
    Kunai
    VERSION 2.0
    Description "Static binary analysis framework wanna-be for different executable file formats."
    PROJECT_HOMEPAGE_URL https://github.com/Fare9/KUNAI-static-analyzer
    )

# Specify the C++ standard required
# in case of KUNAI we use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# use of external library
add_subdirectory(external/zip)
target_link_libraries(Kunai PUBLIC libzip.so)


target_include_directories(Kunai PUBLIC
                           "${PROJECT_SOURCE_DIR}/src/include"
                           "${PROJECT_SOURCE_DIR}/external/zip/src"
                           )