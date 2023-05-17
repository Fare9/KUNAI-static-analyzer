//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file header.cpp
#include "Kunai/DEX/parser/header.hpp"

using namespace KUNAI::DEX;

void Header::parse_headers(stream::KunaiStream *stream)
{
    auto logger = LOGGER::logger();
    auto f_size = stream->get_size();

    logger->debug("header.cpp: start parsing");

    // read the dex header
    stream->read_data<dexheader_t>(dexheader, sizeof(dexheader_t));

    if (dexheader.link_off > f_size)
        throw exceptions::ParserException("Error 'link_off' > file_size");

    if (dexheader.map_off > f_size)
        throw exceptions::ParserException("Error 'map_off' > file_size");

    if (dexheader.string_ids_off > f_size)
        throw exceptions::ParserException("Error 'string_ids_off' > file_size");

    if (dexheader.type_ids_off > f_size)
        throw exceptions::ParserException("Error 'type_ids_off' > file_size");

    if (dexheader.proto_ids_off > f_size)
        throw exceptions::ParserException("Error 'proto_ids_off' > file_size");

    if (dexheader.method_ids_off > f_size)
        throw exceptions::ParserException("Error 'methods_ids_off' > file_size");

    if (dexheader.class_defs_off > f_size)
        throw exceptions::ParserException("Error 'class_defs_off' > file_size");

    if (dexheader.data_off > f_size)
        throw exceptions::ParserException("Error 'data_off' > file_size");

    logger->debug("header.cpp: dex header correctly parsed");
}

void Header::to_xml(std::ofstream &fos)
{
    size_t i;

    fos << std::hex;
    fos << "<header>\n";
    fos << "\t<magic>";
    for (i = 0; i < 8; ++i)
        fos << dexheader.magic[i] << " ";
    fos << "</magic>\n";
    fos << "\t<checksum>" << dexheader.checksum << "</checksum>\n";
    fos << "\t<signature>";
    for (i = 0; i < 20; i++)
        fos << dexheader.signature[i] << " ";
    fos << "</signature>\n";
    fos << "\t<file_size>" << dexheader.file_size << "</file_size>\n";
    fos << "\t<header_size>" << dexheader.header_size << "</header_size>\n";
    fos << "\t<endian_tag>" << dexheader.endian_tag << "</endian_tag>\n";
    fos << "\t<link_size>" << dexheader.link_size << "</link_size>\n";
    fos << "\t<link_offset>" << dexheader.link_off << "</link_offset>\n";
    fos << "\t<map_offset>" << dexheader.map_off << "</map_offset>\n";
    fos << "\t<string_ids_size>" << dexheader.string_ids_size << "</string_ids_size>\n";
    fos << "\t<string_ids_offset>" << dexheader.string_ids_off << "</string_ids_offset>\n";
    fos << "\t<type_ids_size>" << dexheader.type_ids_size << "</type_ids_size>\n";
    fos << "\t<type_ids_offset>" << dexheader.type_ids_off << "</type_ids_offset>\n";
    fos << "\t<proto_ids_size>" << dexheader.proto_ids_size << "</proto_ids_size>\n";
    fos << "\t<proto_ids_offset>" << dexheader.proto_ids_off << "</proto_ids_offset>\n";
    fos << "\t<field_ids_size>" << dexheader.field_ids_size << "</field_ids_size>\n";
    fos << "\t<field_ids_offset>" << dexheader.field_ids_off << "</field_ids_offset>\n";
    fos << "\t<method_ids_size>" << dexheader.method_ids_size << "</method_ids_size>\n";
    fos << "\t<<method_ids_offset>" << dexheader.method_ids_off << "</method_ids_offset>\n";
    fos << "\t<class_defs_size>" << dexheader.class_defs_size << "</class_defs_size>\n";
    fos << "\t<class_defs_offset>" << dexheader.class_defs_off << "</class_defs_offset>\n";
    fos << "\t<data_size>" << dexheader.data_size << "</data_size>\n";
    fos << "\t<data_offset>" << dexheader.data_off << "</data_offset>\n";
}

namespace KUNAI
{
namespace DEX
{
    std::ostream &operator<<(std::ostream &os, const Header &entry)
    {
        size_t i = 0;

        auto &dex_struct = entry.get_dex_header_const();

        os << "DEX Header\n";
        os << "Magic: ";

        for (i = 0; i < 8; ++i)
        {
            os << static_cast<int>(dex_struct.magic[i]);
            if (dex_struct.magic[i] == 0xA)
                os << "(\\n)";
            else if (isprint(dex_struct.magic[i]))
                os << "(" << static_cast<char>(dex_struct.magic[i]) << ")";
            os << " ";
        }
        os << "\n";

        os << std::setw(30) << std::left << std::setfill(' ') << "Checksum: " << dex_struct.checksum << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Signature: ";
        for (i = 0; i < 20; ++i)
            os << static_cast<int>(dex_struct.signature[i]) << " ";
        os << "\n";

        os << std::setw(30) << std::left << std::setfill(' ') << "File Size: " << dex_struct.file_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Header Size: " << dex_struct.header_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Endian Tag: " << dex_struct.endian_tag << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Link Size: " << dex_struct.link_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Link Offset: " << dex_struct.link_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Map Offset: " << dex_struct.map_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "String Ids Size: " << dex_struct.string_ids_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "String Ids Offset: " << dex_struct.string_ids_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Type Ids Size: " << dex_struct.type_ids_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Type Ids Offset: " << dex_struct.type_ids_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Proto Ids Size: " << dex_struct.proto_ids_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Proto Ids Offset: " << dex_struct.proto_ids_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Field Ids Size: " << dex_struct.field_ids_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Field Ids Offset: " << dex_struct.field_ids_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Method Ids Size: " << dex_struct.method_ids_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Method Ids Offset: " << dex_struct.method_ids_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Class Defs Size: " << dex_struct.class_defs_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Class Defs Offset: " << dex_struct.class_defs_off << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Data Size: " << dex_struct.data_size << "\n";
        os << std::setw(30) << std::left << std::setfill(' ') << "Data Offset: " << dex_struct.data_off << "\n";
        return os;
    }
}
}