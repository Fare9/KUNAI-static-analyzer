add_library(Lifter
MjolnIRLifter.cpp
InstructionLifter.cpp
)

target_link_libraries(Lifter PUBLIC
  MjolnIR
  MLIRControlFlowDialect
)