configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-ir-analysis.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-ir-analysis.inc
)

if(USE_MJOLNIR)
    add_executable(test-ir-analysis
        ${CMAKE_CURRENT_SOURCE_DIR}/test-ir-analysis.cpp
    )

    target_link_libraries(test-ir-analysis
        PRIVATE
        
        KUNAI
        MjolnIR
        Lifter
        IRManager
        MjolnIRTransforms
        
        MLIRPass
        MLIRParser
        MLIRTransforms

        ${LLVM_COMMON_LIBS}
        )


    add_test(NAME test-ir-analysis
        COMMAND test-ir-analysis)
endif()