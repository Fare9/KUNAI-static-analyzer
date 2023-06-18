cmake_minimum_required(VERSION 3.5)

project(examples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(KUNAI REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    header-simple-printer
    header-simple-printer.cpp
)


target_link_libraries(header-simple-printer PRIVATE Threads::Threads)

target_link_libraries(header-simple-printer PRIVATE KUNAI)

target_include_directories(header-simple-printer PRIVATE ${KUNAI_INCLUDE_DIRS})

add_executable(
    method-extractor
    method-extractor.cpp
)

target_link_libraries(method-extractor PRIVATE Threads::Threads)

target_link_libraries(method-extractor PRIVATE KUNAI)

target_include_directories(method-extractor PRIVATE ${KUNAI_INCLUDE_DIRS})