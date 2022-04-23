/***
 * @file dex_parser.hpp
 * @author Farenain
 *
 * @brief DEX parser with the different parts of Dalvik File.
 *
 * DEX parser with the different classes parts of Dalvik File structures
 * here we will have a public method to start parsing.
 */

#ifndef DEX_PARSER_HPP
#define DEX_PARSER_HPP

#include <iostream>
#include <memory>
#include <fstream>
#include <cstring>

#include "exceptions.hpp"
#include "dex_header.hpp"
#include "dex_strings.hpp"
#include "dex_types.hpp"
#include "dex_protos.hpp"
#include "dex_fields.hpp"
#include "dex_methods.hpp"
#include "dex_classes.hpp"

#include "dex_dvm_types.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DexParser;

        /**
         * @brief shared_ptr type of DexParser, this is the parsing of the
         * DEX headers, different objects represent the different parts of
         * the DEX.
         */
        using dexparser_t = std::shared_ptr<DexParser>;

        class DexParser
        {
        public:
            DexParser();
            ~DexParser() = default;

            void parse_dex_file(std::ifstream &input_file, std::uint64_t file_size);

            // getter methods for the parser, for getting
            // all the parsed fields.
            dexheader_t get_header()
            {
                return dex_header;
            }

            dexstrings_t get_strings()
            {
                return dex_strings;
            }

            dextypes_t get_types()
            {
                return dex_types;
            }

            dexprotos_t get_protos()
            {
                return dex_protos;
            }

            dexfields_t get_fields()
            {
                return dex_fields;
            }

            dexmethods_t get_methods()
            {
                return dex_methods;
            }

            dexclasses_t get_classes()
            {
                return dex_classes;
            }

            // utilities for analysts to get more specific
            // information.

            // header version
            std::uint32_t get_header_version();
            std::string get_header_version_str();

            // dex api version (if analyzed the AndroidManifest.xml)
            void set_api_version(std::uint32_t api_version)
            {
                this->api_version = api_version;
            }

            std::uint32_t get_api_version()
            {
                return api_version;
            }

            // get format type
            std::string get_format_type()
            {
                return "DEX";
            }

            // get all the ClassesDef from DexClasses
            std::vector<classdef_t> get_classes_def_item();
            std::vector<MethodID *> get_methods_id_item();
            std::vector<FieldID *> get_fields_id_item();
            std::vector<codeitemstruct_t> get_codes_item();
            std::vector<std::string> get_string_values();

            friend std::ostream &operator<<(std::ostream &os, const DexParser &entry);

        private:
            std::uint32_t api_version;

            dexheader_t dex_header = nullptr;
            dexstrings_t dex_strings = nullptr;
            dextypes_t dex_types = nullptr;
            dexprotos_t dex_protos = nullptr;
            dexfields_t dex_fields = nullptr;
            dexmethods_t dex_methods = nullptr;
            dexclasses_t dex_classes = nullptr;
        };
    }
}

#endif