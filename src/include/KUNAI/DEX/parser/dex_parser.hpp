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

#include "KUNAI/Exceptions/exceptions.hpp"
#include "KUNAI/DEX/parser/dex_header.hpp"
#include "KUNAI/DEX/parser/dex_strings.hpp"
#include "KUNAI/DEX/parser/dex_types.hpp"
#include "KUNAI/DEX/parser/dex_protos.hpp"
#include "KUNAI/DEX/parser/dex_fields.hpp"
#include "KUNAI/DEX/parser/dex_methods.hpp"
#include "KUNAI/DEX/parser/dex_classes.hpp"

#include "KUNAI/DEX/DVM/dex_dvm_types.hpp"

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

            /**
             * @brief Parse the given DEX file and extract all the different types from the
             *        header.
             * 
             * @param input_file 
             * @param file_size 
             */
            void parse_dex_file(std::ifstream &input_file, std::uint64_t file_size);

            // getter methods for the parser, for getting
            // all the parsed fields.

            /**
             * @brief Get the header object
             * 
             * @return const DexHeader* 
             */
            const DexHeader* get_header() const
            {
                return dex_header.get();
            }

            /**
             * @brief Get the strings object
             * 
             * @return const DexStrings* 
             */
            const DexStrings* get_strings() const
            {
                return dex_strings.get();
            }

            /**
             * @brief Get the types object
             * 
             * @return const DexTypes* 
             */
            const DexTypes* get_types() const
            {
                return dex_types.get();
            }

            /**
             * @brief Get the protos object
             * 
             * @return dexprotos_t& 
             */
            const DexProtos* get_protos() const
            {
                return dex_protos.get();
            }

            /**
             * @brief Get the fields object.
             * 
             * @return dexfields_t& 
             */
            dexfields_t& get_fields()
            {
                return dex_fields;
            }

            /**
             * @brief Get the methods object.
             * 
             * @return dexmethods_t& 
             */
            dexmethods_t& get_methods()
            {
                return dex_methods;
            }

            /**
             * @brief Get the classes object.
             * 
             * @return dexclasses_t& 
             */
            dexclasses_t& get_classes()
            {
                return dex_classes;
            }

            // utilities for analysts to get more specific
            // information.

            // header version

            /**
             * @brief Get the header version as an unsgned integer.
             * 
             * @return std::uint32_t 
             */
            std::uint32_t get_header_version();


            /**
             * @brief Get the header version as a string.
             * 
             * @return std::string 
             */
            std::string get_header_version_str();

            // dex api version (if analyzed the AndroidManifest.xml)
            void set_api_version(std::uint32_t api_version)
            {
                this->api_version = api_version;
            }

            /**
             * @brief Get the api version from the DEX file.
             * 
             * @return std::uint32_t 
             */
            std::uint32_t get_api_version()
            {
                return api_version;
            }

            /**
             * @brief Get the format type as a string.
             * 
             * @return std::string 
             */
            std::string get_format_type()
            {
                return "DEX";
            }

            /**
             * @brief Get the classes def item object
             * 
             * @return const std::vector<classdef_t>& 
             */
            const std::vector<classdef_t>& get_classes_def_item() const
            {
                return dex_classes->get_classes();
            }

            /**
             * @brief Get the methods id item object.
             * 
             * @return std::vector<methodid_t>& 
             */
            std::vector<methodid_t>& get_methods_id_item() const
            {
                return dex_methods->get_method_ids();
            }
            
            /**
             * @brief Get the fields id item object.
             * 
             * @return std::vector<fieldid_t>& 
             */
            std::vector<fieldid_t>& get_fields_id_item() const
            {
                return dex_fields->get_fields();
            }

            /**
             * @brief Get the codes item object.
             * 
             * @return std::vector<codeitemstruct_t> 
             */
            std::vector<codeitemstruct_t> get_codes_item();

            /**
             * @brief Get the string values object
             * 
             * @return std::vector<std::string> 
             */
            std::vector<std::string> get_string_values();

            const std::vector<encodedfield_t>& get_encoded_fields_from_classes() const
            {
                return encoded_fields;
            }

            friend std::ostream &operator<<(std::ostream &os, const DexParser &entry);

        private:

            /**
             * @brief Method to retrieve from the classes all
             *        the encoded_fields, this can be used
             *        later to improve the performance in the analysis
             *        of the cross references.
             */
            void retrieve_encoded_fields_from_classes();

            std::uint32_t api_version;

            dexheader_t dex_header = nullptr;
            dexstrings_t dex_strings = nullptr;
            dextypes_t dex_types = nullptr;
            dexprotos_t dex_protos = nullptr;
            dexfields_t dex_fields = nullptr;
            dexmethods_t dex_methods = nullptr;
            dexclasses_t dex_classes = nullptr;

            std::vector<encodedfield_t> encoded_fields;
        };
    }
}

#endif