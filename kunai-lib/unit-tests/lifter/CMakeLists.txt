configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test-lifter.in
    ${CMAKE_CURRENT_SOURCE_DIR}/test-lifter.inc
)

if(USE_MJOLNIR)
    add_executable(test-lifter
        ${CMAKE_CURRENT_SOURCE_DIR}/test-lifter.cpp
    )

    target_link_libraries(test-lifter
        PRIVATE
        KUNAI
        MjolnIR
        Lifter

        
        MLIRPass
        MLIRParser
        )

    add_test(NAME test-lifter
        COMMAND test-lifter)
endif()