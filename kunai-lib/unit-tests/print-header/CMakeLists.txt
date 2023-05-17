configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-print-header.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-print-header.inc
)

add_executable(test-print-header
${CMAKE_CURRENT_SOURCE_DIR}/test-print-header.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(test-print-header spdlog zip)

add_test(NAME test-print-header
         COMMAND test-print-header)