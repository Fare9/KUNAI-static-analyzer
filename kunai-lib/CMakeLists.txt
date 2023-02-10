cmake_minimum_required(VERSION 3.5)

# Information from Kunai project
project(
    Kunai
    LANGUAGES CXX
    VERSION 2.5.0
    DESCRIPTION "Tool for static analysis of DEX files with the chance to lift and analyze with an Intermediate Representation"
    HOMEPAGE_URL https://github.com/Fare9/KUNAI-static-analyzer
    )

# Specify the standard required
# we will use C++17 to use things
# like std::optional, std::variant...
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# information about the compiler loaded
# its ID and version
if (CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set, setting to Release")
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# for debug configuration set specific flags
if (NOT WIN32)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O0 -DNDEBUG")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
endif()

# include the externals
add_subdirectory(externals)

message(STATUS "Adding the source files")

message(STATUS "${KUNAI_UTILS_SRC}")
message(STATUS "${KUNAI_DEX_SRC}")

# Kunai Source Definition
set(KUNAI_LIB_SOURCE_FILES
${KUNAI_UTILS_SRC}
${KUNAI_DEX_SRC}
)

message(STATUS "Generating Kunai shared library")

set_target_properties(spdlog
PROPERTIES
POSITION_INDEPENDENT_CODE 1
)

set_target_properties(zip
PROPERTIES
POSITION_INDEPENDENT_CODE 1
)

message(STATUS "Generating object files")

# compile the object files
add_library(kunai-objs
OBJECT
)

# include the CMAKEs from each 
add_subdirectory(lib)

# include directories for Kunai
include_directories(BEFORE
    #${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(kunai-objs
PROPERTIES
POSITION_INDEPENDENT_CODE 1
)


# compile the shared library
add_library(kunai-shared SHARED
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(kunai-shared
spdlog
zip
)

# target properties for shared object
set_target_properties(kunai-shared
PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "kunai"
)

if (UNIT_TESTING)
    add_subdirectory(unit-tests)
endif ()