//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dex.cpp
#include "Kunai/DEX/dex.hpp"

using namespace KUNAI::DEX;

void Dex::initialization(std::string& dex_file_path)
{
    dex_file.open(dex_file_path, std::ifstream::binary);
    
    kunai_stream = std::make_unique<stream::KunaiStream>(dex_file);
}