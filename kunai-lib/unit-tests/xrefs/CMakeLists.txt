configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-xrefs.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-xrefs.inc
)

add_executable(test-xrefs
${CMAKE_CURRENT_SOURCE_DIR}/test-xrefs.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(test-xrefs spdlog zip)

add_test(NAME test-xrefs
         COMMAND test-xrefs)