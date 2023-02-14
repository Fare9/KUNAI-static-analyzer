//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file parser.cpp

#include "Kunai/DEX/parser/parser.hpp"
#include "Kunai/DEX/DVM/dvm_types.hpp"
#include "Kunai/Exceptions/parser_exception.hpp"
#include "Kunai/Exceptions/incorrectdexfile_exception.hpp"

using namespace KUNAI::DEX;

void Parser::parse_file()
{
    std::uint8_t magic[4];
    auto logger = LOGGER::logger();

    logger->debug("parser.cpp: started parsing of dex file");

    if (stream->get_size() < sizeof(Header::dexheader_t))
        throw exceptions::ParserException("parser.cpp: file has incorrect size");
    
    // read a header to do simple check
    stream->read_data<std::uint8_t[4]>(magic, sizeof(std::uint8_t[4]));

    if (memcmp(magic, KUNAI::DEX::dex_magic, 4))
        throw exceptions::IncorrectDexFileException("parser.cpp: file is not a dex file");
    
    // move to the beginning
    stream->seekg(0, std::ios_base::beg);

    // start now parsing
    header.parse_headers(stream);
    
    auto & dex_header = header.get_dex_header_const();

    strings.parse_strings(dex_header.string_ids_off, dex_header.string_ids_size, stream);
    types.parse_types(stream, &strings, dex_header.type_ids_size, dex_header.type_ids_off);
    protos.parse_protos(stream, dex_header.proto_ids_size, dex_header.proto_ids_off, &strings, &types);
    fields.parse_fields(stream, &types, &strings, dex_header.field_ids_off, dex_header.field_ids_size);

    logger->debug("parser.cpp: dex file parsing correct");
}