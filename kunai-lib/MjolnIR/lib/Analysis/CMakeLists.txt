add_library(IRManager
  MjolnIRManager.cpp
)

target_link_libraries(IRManager PUBLIC
  MjolnIR
)