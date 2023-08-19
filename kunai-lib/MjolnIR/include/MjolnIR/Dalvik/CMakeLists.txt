set(LLVM_TARGET_DEFINITIONS MjolnIROps.td)
mlir_tablegen(MjolnIREnums.h.inc -gen-enum-decls)
mlir_tablegen(MjolnIREnums.cpp.inc -gen-enum-defs)

add_mlir_dialect(MjolnIROps MjolnIR)
add_mlir_doc(MjolnIRDialect MjolnIRDialect Dalvik/ -gen-dialect-doc)
add_mlir_doc(MjolnIROps MjolnIROps Dalvik/ -gen-op-doc)