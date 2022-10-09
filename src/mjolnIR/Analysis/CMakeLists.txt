set(KUNAI_MJOLNIR_ANALYSIS_SRC
${CMAKE_CURRENT_LIST_DIR}/ir_graph_ssa.cpp
${CMAKE_CURRENT_LIST_DIR}/optimizer.cpp
${CMAKE_CURRENT_LIST_DIR}/reachingDefinition.cpp
${CMAKE_CURRENT_LIST_DIR}/single_block_optimizations.cpp
${CMAKE_CURRENT_LIST_DIR}/single_instruction_optimizations.cpp
)

set(KUNAI_MJOLNIR_ANALYSIS_INC_FILES
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/Analysis/ir_graph_ssa.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/Analysis/optimizer.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/Analysis/reachingDefinition.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/Analysis/single_block_optimizations.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/mjolnIR/Analysis/single_instruction_optimizations.hpp"
)