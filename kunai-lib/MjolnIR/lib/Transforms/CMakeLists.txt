
add_mlir_library(MjolnIRTransforms
MjolnIRToOpGraph.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_BINARY_DIR}/MjolnIR/include/
${PROJECT_BINARY_DIR}/MjolnIR/include/Dalvik

LINK_LIBS PUBLIC
MLIRAnalysis
MLIRPass
MLIRTransforms
MLIRSupport
)

