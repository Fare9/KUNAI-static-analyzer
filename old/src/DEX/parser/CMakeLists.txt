# DEX parser files
set(KUNAI_DEX_PARSER_SRC
${CMAKE_CURRENT_LIST_DIR}/dex_annotations.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_classes.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_encoded.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_fields.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_header.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_methods.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_parser.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_protos.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_strings.cpp
${CMAKE_CURRENT_LIST_DIR}/dex_types.cpp
)

set(KUNAI_DEX_PARSER_INC_FILES
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_annotations.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_classes.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_encoded.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_fields.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_header.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_methods.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_parser.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_protos.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_strings.hpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/include/KUNAI/DEX/parser/dex_types.hpp"
)