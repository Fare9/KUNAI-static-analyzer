//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-print-header.cpp
// @brief Unit test script for the different header printers.

#include "test-print-header.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"

int main()
{
    std::string dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-assignment-arith-logic/Main.dex";

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    auto parser = dex->get_parser();

    auto &dex_header = parser->get_header();

    std::cout << dex_header << "\n\n";

    auto& dex_strings = parser->get_strings();

    std::cout << dex_strings << "\n\n";

    auto& types = parser->get_types();

    std::cout << types << "\n\n";

    auto& protos = parser->get_protos();

    std::cout << protos << "\n\n";

    auto& fields = parser->get_fields();

    std::cout << fields << "\n\n";

    auto& methods = parser->get_methods();

    std::cout << methods << "\n\n";

    auto& classes = parser->get_classes();

    std::cout << classes << "\n\n";

    return 0;
}