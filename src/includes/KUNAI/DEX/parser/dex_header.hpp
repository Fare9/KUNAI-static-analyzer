/**
 *  @file dex_header.hpp
 *  @author @Farenain
 * 
 *  @brief Class that represents and contains the DEX file header structure.
 * 
 *  Header of DEX file:
 *      DexfileHeader{
 *          ubyte[8] magic,
 *          int checksum,
 *          ubyte[20] signature,
 *          uint file_size,
 *          uint header_size,
 *          uint endian_tag,
 *          uint link_size,
 *          uint link_off,
 *          uint map_off,
 *          uint string_ids_size,
 *          uint string_ids_off,
 *          uint type_ids_size,
 *          uint type_ids_off,
 *          uint proto_ids_size,
 *          uint proto_ids_off,
 *          uint field_ids_size,
 *          uint field_ids_off,
 *          uint method_ids_size,
 *          uint method_ids_off,
 *          uint class_defs_size,
 *          uint class_defs_off,
 *          uint data_size,
 *          uint data_off
 *      }
 * 
 * header header_item
 */

#pragma once

#ifndef DEX_HEADER_HPP
#define DEX_HEADER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "exceptions.hpp"
#include "utils.hpp"

namespace KUNAI {
    namespace DEX {

        class DexHeader
        {
        public:

#pragma pack(1)
            struct dexheader_t {
                std::uint8_t    magic[8];
                std::int32_t    checksum;
                std::uint8_t    signature[20];
                std::uint32_t   file_size;
                std::uint32_t   header_size;
                std::uint32_t   endian_tag;
                std::uint32_t   link_size;
                std::uint32_t   link_off;
                std::uint32_t   map_off;
                std::uint32_t   string_ids_size;
                std::uint32_t   string_ids_off;
                std::uint32_t   type_ids_size;
                std::uint32_t   type_ids_off;
                std::uint32_t   proto_ids_size;
                std::uint32_t   proto_ids_off;
                std::uint32_t   field_ids_size;
                std::uint32_t   field_ids_off;
                std::uint32_t   method_ids_size;
                std::uint32_t   method_ids_off;
                std::uint32_t   class_defs_size;
                std::uint32_t   class_defs_off;
                std::uint32_t   data_size;
                std::uint32_t   data_off;
            };
#pragma pack()

            DexHeader(std::ifstream& input_file, std::uint64_t file_size);
            ~DexHeader();
            
            dexheader_t& get_dex_header();
            std::uint64_t get_dex_header_size();

            /*
            * utilities and printers
            */
            friend std::ostream& operator<<(std::ostream& os, const DexHeader& entry);
            friend std::fstream& operator<<(std::fstream& fos, const DexHeader& entry);
        private:
            struct dexheader_t dex_struct;
        };
    }
}

#endif
