//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file header.hpp
// @brief File to manage the structure of the DEX header
// this will be useful for the other classes that will use
// the values from this header
#ifndef KUNAI_DEX_PARSER_HEADER_HPP
#define KUNAI_DEX_PARSER_HEADER_HPP


#include "Kunai/Utils/logger.hpp"
#include "Kunai/Utils/kunaistream.hpp"
#include "Kunai/Exceptions/parser_exception.hpp"
#include <iostream>
#include <iomanip>

namespace KUNAI
{
    namespace DEX
    {
        class Header
        {
        public:
#pragma pack(1)
            /// @brief Structure with the definition of the DEX header
            /// all these values are later used for parsing the other
            /// headers from DEX
            struct dexheader_t
            {
                std::uint8_t magic[8];          //! magic bytes from dex, different values are possible
                std::int32_t checksum;          //! checksum to see if file is correct
                std::uint8_t signature[20];     //! signature of dex
                std::uint32_t file_size;        //! current file size
                std::uint32_t header_size;      //! size of this header
                std::uint32_t endian_tag;       //! type of endianess of the file
                std::uint32_t link_size;        //! data for statically linked files
                std::uint32_t link_off;         //! 
                std::uint32_t map_off;          //! 
                std::uint32_t string_ids_size;  //! number of strings
                std::uint32_t string_ids_off;   //! offset of the strings
                std::uint32_t type_ids_size;    //! number of types
                std::uint32_t type_ids_off;     //! offset of the types
                std::uint32_t proto_ids_size;   //! number of prototypes
                std::uint32_t proto_ids_off;    //! offset of the prototypes
                std::uint32_t field_ids_size;   //! number of fields
                std::uint32_t field_ids_off;    //! offset of the fields
                std::uint32_t method_ids_size;  //! number of methods
                std::uint32_t method_ids_off;   //! offset of the methods
                std::uint32_t class_defs_size;  //! number of class definitions
                std::uint32_t class_defs_off;   //! offset of the class definitions
                std::uint32_t data_size;        //! data area, containing all the support data for the tables listed above
                std::uint32_t data_off;         //!
            };
#pragma pack()

        private:
            /// @brief Variable that will hold the structure
            struct dexheader_t dexheader;

            /// @brief Internal function for parsing the dex headers
            /// @param stream stream with the dex file
            void parse_headers(stream::KunaiStream* stream);

        public:
            /// @brief DEX header constructor
            /// @param stream stream with the dex file
            Header(stream::KunaiStream* stream)
            {
                parse_headers(stream);
            }

            /// @brief Copy constructor of dex header
            /// @param header reference to the header where to copy from
            Header(Header& header)
            {
                memcpy(&dexheader, &header.dexheader, sizeof(dexheader_t));
            }

            /// @brief destructor of DEX header
            ~Header() = default;

            /// @brief Obtain a constant reference of the dex header struct
            /// if no value will be modified, use this function, it will 
            /// be faster :)
            /// @return const reference to header structure
            const dexheader_t &get_dex_header_const() const
            {
                return dexheader;
            }

            /// @brief Obtain a reference of the dex header struct
            /// just in case in the future DEX modification is allowed
            /// @return reference to header structure
            dexheader_t &get_dex_header()
            {
                return dexheader;
            }

            /// @brief Obtain the size of the dex header structure
            /// @return 
            std::uint64_t get_dex_header_size() const
            {
                return sizeof(dexheader_t);
            }

            /// @brief Pretty printer for the operator << of the DEX header
            /// @param os output stream to print the dex header
            /// @param entry entry to print
            /// @return received output stream
            friend std::ostream &operator<<(std::ostream &os, const Header &entry);

            /// @brief Dump the content of the DEX header to an XML file
            /// @param fos file where to store the string
            void to_xml(std::ofstream &fos);
        };
    } // namespace DEX
} // namespace KUNAI

#endif