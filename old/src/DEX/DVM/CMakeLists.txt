# DEX DVM files
set(KUNAI_DEX_DVM_SRC
${CMAKE_CURRENT_LIST_DIR}/dex_dalvik_opcodes.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_disassembler.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_external_classes.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_external_methods.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_instructions.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_linear_sweep_disassembly.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_recursive_traversal_disassembly.cpp
)

set(KUNAI_DEX_DVM_INC_FILES
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_dalvik_opcodes.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_disassembler.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_dvm_types.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM//dex_external_classes.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_external_methods.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_instructions.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_linear_sweep_disassembly.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/DVM/dex_recursive_traversal_disassembly.hpp"
)

