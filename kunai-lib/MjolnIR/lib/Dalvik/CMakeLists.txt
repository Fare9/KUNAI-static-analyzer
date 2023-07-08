add_mlir_dialect_library(MjolnIR
    MjolnIRTypes.cpp
    MjolnIRDialect.cpp
    MjolnIROps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_BINARY_DIR}/MjolnIR/include/
    ${PROJECT_BINARY_DIR}/MjolnIR/include/MjolnIR/Dalvik

    DEPENDS
    MLIRMjolnIROpsIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRMathDialect
    MLIRControlFlowInterfaces
    MLIRFuncDialect
    MLIRIR
    MLIRSideEffectInterfaces
)