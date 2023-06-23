cmake_minimum_required(VERSION 3.5)

project(backward-slicing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(KUNAI REQUIRED)
find_package(Threads REQUIRED)

add_library(backward_analysis_lib backward_analysis.cpp backward_analysis.hpp)
add_executable(backward-slicing backward-slicing.cpp)

# needed, I don't know why
target_link_libraries(backward-slicing PRIVATE Threads::Threads)

# Link against the KUNAI library
target_link_libraries(backward-slicing PRIVATE KUNAI)
target_link_libraries(backward-slicing PRIVATE backward_analysis_lib)

# Set the include directories for the target
target_include_directories(backward-slicing PRIVATE ${KUNAI_INCLUDE_DIRS})
