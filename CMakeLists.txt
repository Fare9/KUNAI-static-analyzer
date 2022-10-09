cmake_minimum_required(VERSION 3.5)

# Project name for KUNAI
project(
    Kunai
    LANGUAGES CXX
    VERSION 2.0.0
    DESCRIPTION "Tool for static analysis of DEX files with the chance to lift and analyze with an Intermediate Representation"
    HOMEPAGE_URL https://github.com/Fare9/KUNAI-static-analyzer
    )

# Specify the C++ standard required
# in case of KUNAI we use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# show information about compiler used
if (CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Is the C++ from GNU? ${CMAKE_COMPILER_IS_GNUCXX}")
    message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# check if there's some build type selected
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# for debug configuration set specific flags
if (NOT WIN32)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
endif()

message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")

# include externals

include(${CMAKE_CURRENT_SOURCE_DIR}/external/CMakeLists.txt)

include(${CMAKE_CURRENT_SOURCE_DIR}/src/APK/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/DEX/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/DEX/Analysis/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/DEX/DVM/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/DEX/parser/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/mjolnIR/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/mjolnIR/Analysis/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/mjolnIR/Lifters/CMakeLists.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/CMakeLists.txt)

# Kunai Source Definition
set(KUNAI_LIB_SOURCE_FILES
${KUNAI_APK_SRC}
${KUNAI_DEX_SRC}
${KUNAI_DEX_ANALYSIS_SRC}
${KUNAI_DEX_DVM_SRC}
${KUNAI_DEX_PARSER_SRC}
${KUNAI_MJOLNIR_SRC}
${KUNAI_MJOLNIR_ANALYSIS_SRC}
${KUNAI_MJOLNIR_LIFTERS_SRC}
${KUNAI_UTILS_SRC}
)

message(STATUS "Added all the source files")

set(KUNAI_BIN_SOURCE_FILES
${KUNAI_LIB_SOURCE_FILES}
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

set(KUNAI_LIB_INCLUDE_FILES
${KUNAI_APK_INC_FILES}
${KUNAI_DEX_INC_FILES}
${KUNAI_DEX_ANALYSIS_INC_FILES}
${KUNAI_DEX_DVM_SRC}
${KUNAI_DEX_PARSER_INC_FILES}
${KUNAI_MJOLNIR_INC_FILES}
${KUNAI_MJOLNIR_ANALYSIS_INC_FILES}
${KUNAI_MJOLNIR_LIFTERS_INC_FILES}
${KUNAI_UTILS_INC_FILES}
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/exceptions.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/apkunzip_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/disassembler_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/incorrectdexfile_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/incorrectfieldid_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/incorrectprotoid_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/incorrectstringid_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/incorrecttypeid_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/incorrectvalue_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/invalidinstruction_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/ir_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/lifter_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/outofbound_exception.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/Exceptions/parserreading_exception.hpp"
)

set(KUNAI_LIB_PUBLIC_INCLUDE_FILES
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/"
)

message(STATUS "Added all the headers")

message(STATUS "Generating Kunai shared library")

set_target_properties(spdlog
PROPERTIES
POSITION_INDEPENDENT_CODE 1
)

set_target_properties(zip
PROPERTIES
POSITION_INDEPENDENT_CODE 1
)


add_library(kunai-objs
OBJECT
${KUNAI_LIB_SOURCE_FILES}
${KUNAI_LIB_INCLUDE_FILES}
)

set_target_properties(kunai-objs
PROPERTIES
POSITION_INDEPENDENT_CODE 1
)

target_include_directories(kunai-objs PRIVATE "${KUNAI_LIB_PUBLIC_INCLUDE_FILES}")
target_link_libraries(kunai-objs spdlog zip)


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

add_library(kunai-static STATIC
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(kunai-static
spdlog
zip
)

set_target_properties(kunai-static
PROPERTIES
OUTPUT_NAME "kunai"
)

# now compile Kunai's binary
set(KUNAI_BIN_SRC
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
${KUNAI_LIB_SOURCE_FILES}
)

add_executable(kunai-binary
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(kunai-binary PRIVATE "${KUNAI_LIB_PUBLIC_INCLUDE_FILES}")
target_link_libraries(kunai-binary spdlog zip)

set_target_properties(kunai-binary
PROPERTIES
OUTPUT_NAME "Kunai"
)

# include the CMakeLists.txt from the projects
include(${CMAKE_CURRENT_SOURCE_DIR}/projects/CMakeLists.txt)


# installation
message(STATUS "Project will be installed to ${CMAKE_INSTALL_PREFIX}")

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")

# Report to user
foreach(p LIB INCLUDE)
file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX}/${INSTALL_${p}DIR} _path )
message(STATUS "Installing ${p} components to ${_path}")
unset(_path)
endforeach()

# install the libraries
install(
    TARGETS
        kunai-shared
        kunai-static
    ARCHIVE
        DESTINATION ${INSTALL_LIBDIR}
        COMPONENT lib
    RUNTIME
        DESTINATION ${INSTALL_BINDIR}
        COMPONENT bin
    LIBRARY
        DESTINATION ${INSTALL_LIBDIR}
        COMPONENT lib
)

install(
    DIRECTORY "${KUNAI_LIB_PUBLIC_INCLUDE_FILES}"
    DESTINATION ${INSTALL_INCLUDEDIR}
    COMPONENT headers
    FILES_MATCHING
    REGEX "(.*).(hpp|h|def|inc)$"
)