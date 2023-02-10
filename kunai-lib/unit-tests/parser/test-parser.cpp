//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-parser.cpp
// @brief Unit test script for the DEX parser of Kunai.

#include "Kunai/DEX/dex.hpp"
#include <assert.h>

int
main()
{
    std::string dex_file_path = "";

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    return 0;
}