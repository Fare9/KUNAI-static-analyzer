cmake_minimum_required(VERSION 3.5)

project(backward-slicing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(KUNAI REQUIRED)
find_package(Threads REQUIRED)

set(BACKWARD_SLICING_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
)

set(BACKWARD_SLICING_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backward_analysis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backward-slicing.cpp"
)


include_directories(BEFORE
    ${BACKWARD_SLICING_INCLUDE_FILES}
)

add_executable(backward-slicing
    ${BACKWARD_SLICING_SRC_FILES}
)

# needed, I don't know why
target_link_libraries(backward-slicing PRIVATE Threads::Threads)

# Link against the KUNAI library
target_link_libraries(backward-slicing PRIVATE KUNAI)

# Set the include directories for the target
target_include_directories(backward-slicing PRIVATE ${KUNAI_INCLUDE_DIRS})
