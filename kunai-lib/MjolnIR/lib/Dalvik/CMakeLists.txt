add_mlir_dialect_library(MjolnIR
    MjolnIRDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_BINARY_DIR}/MjolnIR/include/
    ${PROJECT_BINARY_DIR}/MjolnIR/include/Dalvik

    DEPENDS
    MLIRMjolnIROpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)