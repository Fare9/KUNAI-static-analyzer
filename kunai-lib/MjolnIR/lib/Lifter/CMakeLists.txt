add_library(Lifter
  MjolnIRLifter.cpp
  InstructionLifter.cpp
  Instructions/Instruction10t.cpp
  Instructions/Instruction10x.cpp
  Instructions/Instruction11n.cpp
  Instructions/Instruction11x.cpp
  Instructions/Instruction12x.cpp
  Instructions/Instruction20t.cpp
  Instructions/Instruction21c.cpp
  Instructions/Instruction21h.cpp
  Instructions/Instruction21s.cpp
  Instructions/Instruction21t.cpp
  Instructions/Instruction22b.cpp
  Instructions/Instruction22c.cpp
  Instructions/Instruction22s.cpp
  Instructions/Instruction22t.cpp
  Instructions/Instruction22x.cpp
  Instructions/Instruction23x.cpp
  Instructions/Instruction30t.cpp
  Instructions/Instruction31c.cpp
  Instructions/Instruction31i.cpp
  Instructions/Instruction32x.cpp
  Instructions/Instruction35c.cpp
  Instructions/Instruction51l.cpp
)

target_link_libraries(Lifter PUBLIC
  MjolnIR
  MLIRControlFlowDialect
)