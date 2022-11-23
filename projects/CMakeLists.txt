# test binaries
add_executable(test_apk_analysis
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_apk_analysis.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_apk_analysis PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_apk_analysis spdlog zip)

add_executable(test_dex_analysis
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_dex_analysis.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_dex_analysis PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_dex_analysis spdlog zip)

add_executable(test_dex_recursive_analysis
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_dex_recursive_analysis.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_dex_recursive_analysis PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_dex_recursive_analysis spdlog zip)

add_executable(test_dex_disassembler
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_dex_disassembler.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_dex_disassembler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_dex_disassembler spdlog zip)

add_executable(test_dex_lifter
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_dex_lifter.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_dex_lifter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_dex_lifter spdlog zip)

add_executable(test_dex_parser
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_dex_parser.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_dex_parser PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_dex_parser spdlog zip)

add_executable(test_dominators
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_dominators.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_dominators PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_dominators spdlog zip)

add_executable(test_ir_graph
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_ir_graph.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_ir_graph PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_ir_graph spdlog zip)

add_executable(test_ir
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_ir.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_ir PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_ir spdlog zip)

add_executable(test_ssa_form
${CMAKE_CURRENT_SOURCE_DIR}/projects/test_ssa_form.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test_ssa_form PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test_ssa_form spdlog zip)

add_executable(test-optimizations
${CMAKE_CURRENT_SOURCE_DIR}/projects/test-optimizations.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test-optimizations PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test-optimizations spdlog zip)

add_executable(test-optimizations2
${CMAKE_CURRENT_SOURCE_DIR}/projects/test-optimizations2.cpp
$<TARGET_OBJECTS:kunai-objs>
)

target_include_directories(test-optimizations2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/include/")
target_link_libraries(test-optimizations2 spdlog zip)

set_target_properties(test_apk_analysis
test_dex_analysis
test_dex_recursive_analysis
test_dex_disassembler
test_dex_lifter
test_dex_parser
test_dominators
test_ir_graph
test_ir
test_ssa_form
test-optimizations
test-optimizations2
PROPERTIES
ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/project"
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/project"
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/project"
)
