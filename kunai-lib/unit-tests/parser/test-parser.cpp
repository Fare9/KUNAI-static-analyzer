//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-parser.cpp
// @brief Unit test script for the DEX parser of Kunai.

#include "test-parser.inc"
#include "Kunai/DEX/dex.hpp"
#include <assert.h>

void check_header_struct(const KUNAI::DEX::Header::dexheader_t &header_struct)
{
    assert(
        memcmp(header_struct.magic, "dex\n035\0", 8) == 0 &&
        "dex magic not correct");

    assert(
        header_struct.checksum == 0x123ff4f7 &&
        "dex checksum not correct");

    assert(
        (header_struct.signature[0] == 0x7c &&
         header_struct.signature[1] == 0xda &&
         header_struct.signature[18] == 0x84 &&
         header_struct.signature[19] == 0xa6) &&
        "dex signature not correct");

    assert(
        header_struct.file_size == 1768 &&
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
        header_struct.string_ids_size == 39 &&
        "dex string ids size not correct"
    );

    assert(
        header_struct.string_ids_off == 112 &&
        "dex string offset not correct"
    );

    assert(
        header_struct.type_ids_size == 15 &&
        "dex type ids size not correct"
    );

    assert(
        header_struct.type_ids_off == 268 &&
        "dex type offset not correct"
    );

    assert(
        header_struct.proto_ids_size == 9 &&
        "dex proto_ids_size not correct"
    );

    assert(
        header_struct.proto_ids_off == 328 &&
        "dex proto_ids_off not correct"
    );

    assert(
        header_struct.field_ids_size == 4 &&
        "dex field_ids_size not correct"
    );

    assert(
        header_struct.field_ids_off == 436 &&
        "dex field_ids_off not correct"
    );

    assert(
        header_struct.method_ids_size == 12 &&
        "dex method_ids_size not correct"
    );

    assert(
        header_struct.method_ids_off == 468 &&
        "dex method_ids_off not correct"
    );

    assert(
        header_struct.class_defs_size == 1 &&
        "dex class_defs_size not correct"
    );

    assert(
        header_struct.class_defs_off == 564 &&
        "dex class_defs_off not correct"
    );

    assert(
        header_struct.data_size == 1172 &&
        "dex data_size not correct"
    );

    assert(
        header_struct.data_off == 596 &&
        "dex data_off not correct"
    );
}

int main()
{
    std::string dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-assignment-arith-logic/Main.dex";

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    auto parser = dex->get_parser();

    auto &dex_header = parser->get_header_const();

    auto &header_struct = dex_header.get_dex_header_const();

    check_header_struct(header_struct);

    return 0;
}