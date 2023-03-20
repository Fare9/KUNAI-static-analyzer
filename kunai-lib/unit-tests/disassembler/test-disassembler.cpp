//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-disassembler.cpp
// @brief Unit test script for the disassembler of Kunai

#include <iostream>
#include <assert.h>
#include <vector>

#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include "test-disassembler.inc"

std::vector<std::uint8_t> raw_buffer = {
    0x12,0x10,0x46,0x0,0x6,0x0,0x71,0x10,
    0x5,0x0,0x0,0x0,0xc,0x0,0x6e,0x10,0x4,
    0x0,0x0,0x0,0xa,0x0,0x19,0x2,0x0,0x40,
    0x1a,0x1,0x19,0x0,0x13,0x4,0xa,0x0,0xb0,
    0x40,0x19,0x4,0xf0,0x3f,0xce,0x42,0x71,
    0x40,0x2,0x0,0x1,0x32,0xe,0x0
};

std::vector<std::string> expected_result = {
    "const/4 v0, 1",
    "aget-object v0, v6, v0",
    "invoke-static {v0}, java.lang.Integer java.lang.Integer->valueOf(java.lang.String)",
    "move-result-object v0",
    "invoke-virtual {v0}, int java.lang.Integer->intValue()",
    "move-result v0",
    "const-wide/high16 v2, 4611686018427387904",
    "const-string v1, this is a test(25)",
    "const/16 v4, 10",
    "add-int/2addr v4, v0",
    "const-wide/high16 v4, 4607182418800017408",
    "div-double/2addr v4, v2",
    "invoke-static {v1, v0, v2, v3}, void Main->test(java.lang.String,int,double)",
    "return-void"
};

int main()
{
    std::string dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-disassembler/classes.dex";

    auto logger = KUNAI::LOGGER::logger();

    logger->set_level(spdlog::level::debug);

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
    {
        logger->error("Error parsing the DEX file");
        return -1;
    }

    auto disassembler = dex->get_dex_disassembler();

    disassembler->disassembly_dex();

    if (!disassembler->correct_disassembly())
    {
        logger->error("There was an error while disassembly DEX file");
        return -1;
    }

    auto &methods_instrs = disassembler->get_dex_instructions();

    for (auto &method_instrs : methods_instrs)
    {
        auto encoded_method = method_instrs.first;

        if (encoded_method->getMethodID()->get_name() == "main")
        {
            const auto &instrs = method_instrs.second;

            assert(instrs.size() == expected_result.size() &&
                    "Instructions size mismatch with expected result");

            for (size_t I = 0, E = instrs.size(); I < E; ++I)
            {
                assert(instrs[I]->print_instruction() == expected_result[I]  &&
                        "Instruction doesn't match expected result");
            }

        }
    }

    auto disassembled_instructions = disassembler->disassembly_buffer(raw_buffer);

    for (size_t I = 0, E = disassembled_instructions.size(); I < E; ++I)
    {
        assert(disassembled_instructions[I]->print_instruction() == expected_result[I]  &&
                        "Instruction doesn't match expected result");
    }

    logger->info("test-disassembler passed correctly");

    return 0;
}