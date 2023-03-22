//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer, library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-graph.cpp
// @brief Unit test script for the DEX graph creation

#include "test-graph.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include <assert.h>

std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, std::string>> expected_result = {
    {0, {{0, "new-instance v3, java.util.Scanner (10)"}, {4, "sget-object v0, Ljava/lang/System;->in Ljava/io/InputStream; (0)"}, {8, "invoke-direct {v3, v0}, void java.util.Scanner-><init>(java.io.InputStream)"}, {14, "invoke-virtual {v3}, int java.util.Scanner->nextInt()"}, {20, "move-result v0"}, {22, "nop"}, {24, "nop"}, {26, "if-nez v0, 10"}}},
    {30, {{30, "sget-object v1, Ljava/lang/System;->out Ljava/io/PrintStream; (1)"}, {34, "const-string v2, \"You gave me value 0...\" (18)"}, {38, "invoke-virtual {v1, v2}, void java.io.PrintStream->println(java.lang.String)"}, {44, "goto 8"}}},
    {46, {{46, "sget-object v1, Ljava/lang/System;->out Ljava/io/PrintStream; (1)"}, {50, "const-string v2, \"Not a bad value :D...\" (14)"}, {54, "invoke-virtual {v1, v2}, void java.io.PrintStream->println(java.lang.String)"}}},
    {60, {
             {60, "const/4 v1, 3"},
             {62, "div-int/2addr v1, v0"},
             {64, "sget-object v0, Ljava/lang/System;->out Ljava/io/PrintStream; (1)"},
             {68, "invoke-virtual {v0, v1}, void java.io.PrintStream->println(int)"},
             {74, "goto 9"},
         }},
    {76, {
             {76, "move-exception v0"},
             {78, "sget-object v0, Ljava/lang/System;->out Ljava/io/PrintStream; (1)"},
             {82, "const-string v1, \"Divided by zero operation cannot possible\" (1)"},
             {86, "invoke-virtual {v0, v1}, void java.io.PrintStream->println(java.lang.String)"},
         }},
    {92, {
             {92, "invoke-virtual {v3}, void java.util.Scanner->close()"},
             {98, "return-void"},
         }}};

std::vector<std::string> expected_edges = {
    "BB-Start Block -> BB-0",
    "BB-46 -> BB-60",
    "BB-76 -> BB-92",
    "BB-60 -> BB-92",
    "BB-30 -> BB-60",
    "BB-0 -> BB-30",
    "BB-0 -> BB-46",
    "BB-Start Block -> BB-76",
    "BB-92 -> BB-End Block"
};

int main()
{
    std::string dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-try-catch/Main.dex";

    auto logger = KUNAI::LOGGER::logger();

    logger->set_level(spdlog::level::debug);

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    auto analysis = dex->get_analysis(false);

    const auto &methods = analysis->get_methods();

    for (const auto &method : methods)
    {
        if (method.second->get_name() == "main")
        {
            const auto &blocks = method.second->get_basic_blocks();

            for (const auto block : blocks.get_nodes())
            {
                if (block->is_start_block())
                    std::cout << "BB-Start Block\n";
                else if (block->is_end_block())
                    std::cout << "BB-End Block\n";
                else if (block->is_try_block())
                    std::cout << "BB-" << block->get_first_address() << " (try block)"
                              << "\n";
                else if (block->is_catch_block())
                    std::cout << "BB-" << block->get_first_address() << " (catch block)"
                              << "\n";
                else
                    std::cout << "BB-" << block->get_first_address() << "\n";

                for (const auto &instr : block->get_instructions())
                    std::cout << "\t" << instr->get_address() << " : " << instr->print_instruction() << "\n";

                if (!block->is_start_block() && !block->is_end_block())
                {
                    auto start_block_addr = block->get_first_address();

                    for (const auto &instr : block->get_instructions())
                    {
                        assert(expected_result[start_block_addr][instr->get_address()] == instr->print_instruction() && "Expected instruction different to disassembled one");
                    }
                }
            }

            std::cout << "Edges from graph,\n";

            std::size_t i = 0;

            for (const auto &edges : blocks.get_edges())
            {
                std::stringstream output;
                if (edges.first->is_start_block())
                    output << "BB-Start Block";
                else
                    output << "BB-" << edges.first->get_first_address();

                if (edges.second->is_end_block())
                    output << " -> BB-End Block";
                else
                    output << " -> BB-" << edges.second->get_first_address();

                assert(output.str() == expected_edges[i++] && "Expected edge in the graph incorrect");
            }

            break;
        }
    }
}