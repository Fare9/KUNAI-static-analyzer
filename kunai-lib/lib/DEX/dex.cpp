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
    auto logger = LOGGER::logger();

    dex_file.open(dex_file_path, std::ifstream::binary);
    
    kunai_stream = std::make_unique<stream::KunaiStream>(dex_file);
    parser = std::make_unique<Parser>(kunai_stream.get());

    try
    {
        parser->parse_file();
        parsing_correct = true;
        dex_disassembler = std::make_unique<DexDisassembler>(parser.get());
    }
    catch(const std::exception& e)
    {
        logger->error("dex.cpp: {}", e.what());
        parsing_correct = false;
    }
}

Analysis * Dex::get_analysis(bool create_xrefs)
{
    if (!parsing_correct)
        return nullptr;
    /// disassembly the dex file
    dex_disassembler->disassembly_dex();
    /// check if disassembly was correct
    if (!dex_disassembler->correct_disassembly())
        return nullptr;
    
    analysis = std::make_unique<Analysis>(parser.get(), dex_disassembler.get(), create_xrefs);

    return analysis.get();
}


std::unique_ptr<Dex> Dex::parse_dex_file(std::string& dex_file_path)
{
    return std::make_unique<Dex>(dex_file_path);
}

std::unique_ptr<Dex> Dex::parse_dex_file(char * dex_file_path)
{
    std::string dex_path(dex_file_path);

    return std::make_unique<Dex>(dex_path);
}

std::unique_ptr<Dex> Dex::parse_dex_file(const char * dex_file_path)
{
    std::string dex_path(dex_file_path);

    return std::make_unique<Dex>(dex_path);
}