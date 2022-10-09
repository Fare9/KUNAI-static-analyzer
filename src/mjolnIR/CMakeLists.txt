set(KUNAI_MJOLNIR_SRC
${CMAKE_CURRENT_LIST_DIR}/ir_blocks.cpp
${CMAKE_CURRENT_LIST_DIR}/ir_expr.cpp
${CMAKE_CURRENT_LIST_DIR}/ir_graph.cpp
${CMAKE_CURRENT_LIST_DIR}/ir_stmnt.cpp
${CMAKE_CURRENT_LIST_DIR}/ir_type.cpp
${CMAKE_CURRENT_LIST_DIR}/ir_utils.cpp
)

set(KUNAI_MJOLNIR_INC_FILES
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/ir_grammar.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/ir_graph.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/arch/ir_dalvik.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/arch/ir_x86.hpp"
)