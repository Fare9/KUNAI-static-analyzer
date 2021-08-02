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

        class DexParser
        {
        public:
            DexParser();
            ~DexParser();

            void parse_dex_file(std::ifstream &input_file, std::uint64_t file_size);

            // getter methods for the parser, for getting
            // all the parsed fields.
            std::shared_ptr<DexHeader> get_header();
            std::shared_ptr<DexStrings> get_strings();
            std::shared_ptr<DexTypes> get_types();
            std::shared_ptr<DexProtos> get_protos();
            std::shared_ptr<DexFields> get_fields();
            std::shared_ptr<DexMethods> get_methods();
            std::shared_ptr<DexClasses> get_classes();

            // utilities for analysts to get more specific
            // information.

            // header version
            std::uint32_t get_header_version();
            std::string get_header_version_str();

            // dex api version (if analyzed the AndroidManifest.xml)
            void set_api_version(std::uint32_t api_version);
            std::uint32_t get_api_version();

            // get format type
            std::string get_format_type();

            // get all the ClassesDef from DexClasses
            std::vector<std::shared_ptr<ClassDef>> get_classes_def_item();
            std::vector<MethodID *> get_methods_id_item();
            std::vector<FieldID *> get_fields_id_item();
            std::vector<std::shared_ptr<CodeItemStruct>> get_codes_item();
            std::vector<std::string> get_string_values();

            friend std::ostream &operator<<(std::ostream &os, const DexParser &entry);

        private:
            std::uint32_t api_version;

            std::shared_ptr<DexHeader> dex_header = nullptr;
            std::shared_ptr<DexStrings> dex_strings = nullptr;
            std::shared_ptr<DexTypes> dex_types = nullptr;
            std::shared_ptr<DexProtos> dex_protos = nullptr;
            std::shared_ptr<DexFields> dex_fields = nullptr;
            std::shared_ptr<DexMethods> dex_methods = nullptr;
            std::shared_ptr<DexClasses> dex_classes = nullptr;
        };
    }
}

#endif