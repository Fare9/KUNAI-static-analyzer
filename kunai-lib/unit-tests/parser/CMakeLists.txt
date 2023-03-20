configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-parser.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-parser.inc
)

add_executable(test-parser
${CMAKE_CURRENT_SOURCE_DIR}/test-parser.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(test-parser spdlog zip)

add_test(NAME test-parser
         COMMAND test-parser)