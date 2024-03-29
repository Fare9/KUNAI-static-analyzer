//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-parser.cpp
// @brief Unit test script for the DEX parser of Kunai.

#include "test-parser.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include <assert.h>

void check_header_struct(const KUNAI::DEX::Header::dexheader_t &header_struct)
{
    assert(
        memcmp(header_struct.magic, "dex\n035\0", 8) == 0 &&
        "dex magic not correct");

    assert(
        header_struct.checksum == 0x8df011a &&
        "dex checksum not correct");

    assert(
        (header_struct.signature[0] == 0x36 &&
         header_struct.signature[1] == 0x65 &&
         header_struct.signature[18] == 0x60 &&
         header_struct.signature[19] == 0xad) &&
        "dex signature not correct");

    assert(
        header_struct.file_size == 1876 &&
        "dex file size not correct");

    assert(
        header_struct.header_size == 112 &&
        "dex header size not correct"
    );

    assert(
        header_struct.link_size == 0 &&
        "dex link size not correct"
    );

    assert(
        header_struct.link_off == 0 &&
        "dex link offset not correct"
    );

    assert(
        header_struct.string_ids_size == 41 &&
        "dex string ids size not correct"
    );

    assert(
        header_struct.string_ids_off == 112 &&
        "dex string offset not correct"
    );

    assert(
        header_struct.type_ids_size == 16 &&
        "dex type ids size not correct"
    );

    assert(
        header_struct.type_ids_off == 276 &&
        "dex type offset not correct"
    );

    assert(
        header_struct.proto_ids_size == 9 &&
        "dex proto_ids_size not correct"
    );

    assert(
        header_struct.proto_ids_off == 340 &&
        "dex proto_ids_off not correct"
    );

    assert(
        header_struct.field_ids_size == 5 &&
        "dex field_ids_size not correct"
    );

    assert(
        header_struct.field_ids_off == 448 &&
        "dex field_ids_off not correct"
    );

    assert(
        header_struct.method_ids_size == 12 &&
        "dex method_ids_size not correct"
    );

    assert(
        header_struct.method_ids_off == 488 &&
        "dex method_ids_off not correct"
    );

    assert(
        header_struct.class_defs_size == 1 &&
        "dex class_defs_size not correct"
    );

    assert(
        header_struct.class_defs_off == 584 &&
        "dex class_defs_off not correct"
    );

    assert(
        header_struct.data_size == 1260 &&
        "dex data_size not correct"
    );

    assert(
        header_struct.data_off == 616 &&
        "dex data_off not correct"
    );
}

int main()
{
    std::string dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-assignment-arith-logic/Main.dex";

    auto logger = KUNAI::LOGGER::logger();

    logger->set_level(spdlog::level::debug);

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    auto parser = dex->get_parser();

    auto &dex_header = parser->get_header_const();

    auto &header_struct = dex_header.get_dex_header_const();

    check_header_struct(header_struct);

    return 0;
}