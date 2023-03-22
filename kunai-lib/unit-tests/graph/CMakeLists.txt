configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-graph.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-graph.inc
)

add_executable(test-graph
${CMAKE_CURRENT_SOURCE_DIR}/test-graph.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(test-graph spdlog zip)

add_test(NAME test-graph
         COMMAND test-graph)