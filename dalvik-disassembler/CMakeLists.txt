cmake_minimum_required(VERSION 3.5)

project(dalvik-disassembler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(KUNAI REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    dalvik-disassembler
    dalvik-disassembler.cpp
)

# needed, I don't know why
target_link_libraries(dalvik-disassembler PRIVATE Threads::Threads)

# Link against the KUNAI library
target_link_libraries(dalvik-disassembler PRIVATE KUNAI)

# Set the include directories for the target
target_include_directories(dalvik-disassembler PRIVATE ${KUNAI_INCLUDE_DIRS})