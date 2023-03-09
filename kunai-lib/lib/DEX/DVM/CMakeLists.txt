target_sources(kunai-objs PRIVATE
${CMAKE_CURRENT_LIST_DIR}/dalvik_opcodes.cpp
${CMAKE_CURRENT_LIST_DIR}/dalvik_instructions.cpp
${CMAKE_CURRENT_LIST_DIR}/disassembler.cpp
${CMAKE_CURRENT_LIST_DIR}/linear_sweep_disassembler.cpp
${CMAKE_CURRENT_LIST_DIR}/recursive_traversal_disassembler.cpp
)