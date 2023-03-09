cmake_minimum_required(VERSION 3.5)

include(FetchContent)

set(FETCHCONTENT_QUIET off)

execute_process(
    COMMAND ping www.google.com -c 2
    ERROR_QUIET
    RESULT_VARIABLE NO_CONNECTION
)

if(NOT NO_CONNECTION EQUAL 0)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
    message(WARNING "Fetch offline mode: requires already populated _deps")
else()
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF)
endif()

# ------------------------------------
# Download the Logging utility from spdlog
# repo
# ------------------------------------
FetchContent_Declare(
    spdlog

    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.5.0
)

FetchContent_GetProperties(spdlog)

if(NOT spdlog_POPULATED)
    message(STATUS "Cloning spdlog")
    FetchContent_Populate(spdlog)
    add_subdirectory(
        ${spdlog_SOURCE_DIR}
        ${spdlog_BINARY_DIR}
    )
endif()

# ------------------------------------
# Download the zip project for unpacking
# apks
# ------------------------------------
FetchContent_Declare(
    zip

    GIT_REPOSITORY https://github.com/kuba--/zip.git
)

FetchContent_GetProperties(zip)

if(NOT zip_POPULATED)
    message(STATUS "Cloning zip")
    FetchContent_Populate(zip)
    add_subdirectory(
        ${zip_SOURCE_DIR}
        ${zip_BINARY_DIR}
    )
endif()

FetchContent_MakeAvailable(spdlog zip)