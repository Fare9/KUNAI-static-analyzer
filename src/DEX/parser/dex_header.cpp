#include "dex_header.hpp"

namespace KUNAI
{
    namespace DEX
    {
        DexHeader::DexHeader(std::ifstream &input_file, std::uint64_t file_size)
        {
            if (!KUNAI::read_data_file<DexHeader::dexheader_t>(this->dex_struct, sizeof(DexHeader::dex_struct), input_file))
                throw exceptions::ParserReadingException("Error reading DEX Header");

            if (dex_struct.link_off > file_size)
                throw exceptions::OutOfBoundException("Error 'link_off' out of file bound");

            if (dex_struct.map_off > file_size)
                throw exceptions::OutOfBoundException("Error 'map_off' out of file bound");

            if (dex_struct.string_ids_off > file_size)
                throw exceptions::OutOfBoundException("Error 'string_ids_off' out of file bound");

            if (dex_struct.type_ids_off > file_size)
                throw exceptions::OutOfBoundException("Error 'type_ids_off' out of file bound");

            if (dex_struct.proto_ids_off > file_size)
                throw exceptions::OutOfBoundException("Error 'proto_ids_off' out of file bound");

            if (dex_struct.field_ids_off > file_size)
                throw exceptions::OutOfBoundException("Error 'field_ids_off' out of file bound");

            if (dex_struct.method_ids_off > file_size)
                throw exceptions::OutOfBoundException("Error 'method_ids_off' out of file bound");

            if (dex_struct.class_defs_off > file_size)
                throw exceptions::OutOfBoundException("Error 'class_defs_off' out of file bound");

            if (dex_struct.data_off > file_size)
                throw exceptions::OutOfBoundException("Error 'data_off' out of file bound");
        }

        std::ostream &operator<<(std::ostream &os, const DexHeader &entry)
        {
            size_t i;

            os << std::hex;
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Header ===========" << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Magic: ";
            for (i = 0; i < 8; i++)
            {
                os << static_cast<int>(entry.dex_struct.magic[i]);
                if (entry.dex_struct.magic[i] == 0xa)
                    os << "(\\n)";
                else if (isprint(entry.dex_struct.magic[i]))
                    os << "(" << static_cast<char>(entry.dex_struct.magic[i]) << ")";
                os << " ";
            }
            os << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Checksum: " << entry.dex_struct.checksum << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Signature: ";
            for (i = 0; i < 20; i++)
                os << static_cast<int>(entry.dex_struct.signature[i]) << " ";
            os << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "File Size: " << entry.dex_struct.file_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Header Size: " << entry.dex_struct.header_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Endian Tag: " << entry.dex_struct.endian_tag << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Link Size: " << entry.dex_struct.link_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Link Offset: " << entry.dex_struct.link_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Map Offset: " << entry.dex_struct.map_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "String Ids Size: " << entry.dex_struct.string_ids_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "String Ids Offset: " << entry.dex_struct.string_ids_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Type Ids Size: " << entry.dex_struct.type_ids_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Type Ids Offset: " << entry.dex_struct.type_ids_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Proto Ids Size: " << entry.dex_struct.proto_ids_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Proto Ids Offset: " << entry.dex_struct.proto_ids_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Field Ids Size: " << entry.dex_struct.field_ids_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Field Ids Offset: " << entry.dex_struct.field_ids_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Method Ids Size: " << entry.dex_struct.method_ids_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Method Ids Offset: " << entry.dex_struct.method_ids_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Class Defs Size: " << entry.dex_struct.class_defs_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Class Defs Offset: " << entry.dex_struct.class_defs_off << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Data Size: " << entry.dex_struct.data_size << std::endl;
            os << std::setw(30) << std::left << std::setfill(' ') << "Data Offset: " << entry.dex_struct.data_off << std::endl;
            return os;
        }

        std::fstream &operator<<(std::fstream &fos, const DexHeader &entry)
        {
            size_t i;
            std::stringstream stream;

            stream << std::hex;
            stream << "<header>" << std::endl;
            stream << "\t<magic>";
            for (i = 0; i < 8; i++)
                stream << entry.dex_struct.magic[i] << " ";
            stream << "</magic>" << std::endl;
            stream << "\t<checksum>" << entry.dex_struct.checksum << "</checksum>" << std::endl;
            stream << "\t<signature>";
            for (i = 0; i < 20; i++)
                stream << entry.dex_struct.signature[i] << " ";
            stream << "</signature>" << std::endl;
            stream << "\t<file_size>" << entry.dex_struct.file_size << "</file_size>" << std::endl;
            stream << "\t<header_size>" << entry.dex_struct.header_size << "</header_size>" << std::endl;
            stream << "\t<endian_tag>" << entry.dex_struct.endian_tag << "</endian_tag>" << std::endl;
            stream << "\t<link_size>" << entry.dex_struct.link_size << "</link_size>" << std::endl;
            stream << "\t<link_offset>" << entry.dex_struct.link_off << "</link_offset>" << std::endl;
            stream << "\t<map_offset>" << entry.dex_struct.map_off << "</map_offset>" << std::endl;
            stream << "\t<string_ids_size>" << entry.dex_struct.string_ids_size << "</string_ids_size>" << std::endl;
            stream << "\t<string_ids_offset>" << entry.dex_struct.string_ids_off << "</string_ids_offset>" << std::endl;
            stream << "\t<type_ids_size>" << entry.dex_struct.type_ids_size << "</type_ids_size>" << std::endl;
            stream << "\t<type_ids_offset>" << entry.dex_struct.type_ids_off << "</type_ids_offset>" << std::endl;
            stream << "\t<proto_ids_size>" << entry.dex_struct.proto_ids_size << "</proto_ids_size>" << std::endl;
            stream << "\t<proto_ids_offset>" << entry.dex_struct.proto_ids_off << "</proto_ids_offset>" << std::endl;
            stream << "\t<field_ids_size>" << entry.dex_struct.field_ids_size << "</field_ids_size>" << std::endl;
            stream << "\t<field_ids_offset>" << entry.dex_struct.field_ids_off << "</field_ids_offset>" << std::endl;
            stream << "\t<method_ids_size>" << entry.dex_struct.method_ids_size << "</method_ids_size>" << std::endl;
            stream << "\t<<method_ids_offset>" << entry.dex_struct.method_ids_off << "</method_ids_offset>" << std::endl;
            stream << "\t<class_defs_size>" << entry.dex_struct.class_defs_size << "</class_defs_size>" << std::endl;
            stream << "\t<class_defs_offset>" << entry.dex_struct.class_defs_off << "</class_defs_offset>" << std::endl;
            stream << "\t<data_size>" << entry.dex_struct.data_size << "</data_size>" << std::endl;
            stream << "\t<data_offset>" << entry.dex_struct.data_off << "</data_offset>" << std::endl;

            fos.write(stream.str().c_str(), stream.str().size());

            return fos;
        }
    }
}