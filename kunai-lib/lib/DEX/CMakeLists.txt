target_sources(kunai-objs PRIVATE
${CMAKE_CURRENT_LIST_DIR}/dex.cpp
)

add_subdirectory(parser)
add_subdirectory(DVM)
add_subdirectory(analysis)