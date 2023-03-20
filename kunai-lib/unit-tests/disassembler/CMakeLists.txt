configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-disassembler.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-disassembler.inc
)

add_executable(test-disassembler
${CMAKE_CURRENT_SOURCE_DIR}/test-disassembler.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_link_libraries(test-disassembler spdlog zip)

add_test(NAME test-disassembler
         COMMAND test-disassembler)