add_mlir_dialect_library(MjolnIR
    MjolnIRDialect.cpp
    #MjolnIRTypes.cpp
    #MjolnIROps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_BINARY_DIR}/MjolnIR/include/
    ${PROJECT_BINARY_DIR}/MjolnIR/include/Dalvik

	LINK_LIBS PUBLIC
	MLIRIR
    MLIRInferTypeOpInterface
)