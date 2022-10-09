/***
 * @file dex_methods.hpp
 * @author @Farenain
 *
 * @brief Android methods in the java files of Android.
 *        From these methods we could start applying the
 *        disassembly.
 *
 * MethodIDStruct {
 *     ushort class_idx, # id of class where is the method, it is a type_id
 *     ushort proto_idx, # method prototype with return value and the parameters
 *     uint name_idx # id of string with method name
 * }
 *
 * MethodIDStruct[] dex_methods;
 */

#ifndef DEX_METHODS_HPP
#define DEX_METHODS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <utility>
#include <map>

#include "KUNAI/Exceptions/exceptions.hpp"
#include "KUNAI/Utils/utils.hpp"
#include "KUNAI/DEX/parser/dex_strings.hpp"
#include "KUNAI/DEX/parser/dex_types.hpp"
#include "KUNAI/DEX/parser/dex_protos.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class MethodID;

        using methodid_t = std::shared_ptr<MethodID>;

        class MethodID
        {
        public:
            MethodID(std::uint16_t class_idx,
                     std::uint16_t proto_idx,
                     std::uint32_t name_idx,
                     dexstrings_t& dex_strings,
                     dextypes_t& dex_types,
                     dexprotos_t& dex_protos);

            ~MethodID() = default;

            type_t get_method_class()
            {
                return class_idx.second;
            }

            protoid_t get_method_prototype()
            {
                return proto_idx.second;
            }

            std::string* get_method_name()
            {
                return name_idx.second;
            }

            std::string get_method_str();

            friend std::ostream &operator<<(std::ostream &os, const MethodID &entry);

        private:
            std::pair<std::uint16_t, type_t> class_idx;
            std::pair<std::uint16_t, protoid_t> proto_idx;
            std::pair<std::uint32_t, std::string *> name_idx;
        };

        class DexMethods;

        using dexmethods_t = std::shared_ptr<DexMethods>;

        class DexMethods
        {
        public:
            DexMethods(std::ifstream &input_file,
                       std::uint32_t number_of_methods,
                       std::uint32_t offset,
                       dexstrings_t& dex_strings,
                       dextypes_t& dex_types,
                       dexprotos_t& dex_protos);

            ~DexMethods() = default;

            std::uint32_t get_number_of_methods()
            {
                return number_of_methods;
            }

            std::vector<methodid_t>& get_method_ids()
            {
                return method_ids;
            }

            methodid_t get_method_by_order(size_t pos);

            friend std::ostream &operator<<(std::ostream &os, const DexMethods &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexMethods &entry);

        private:
            bool parse_methods(std::ifstream &input_file);

            std::uint32_t number_of_methods;
            std::uint32_t offset;
            dexstrings_t& dex_strings;
            dextypes_t& dex_types;
            dexprotos_t& dex_protos;

            std::vector<methodid_t> method_ids;
        };
    }
}

#endif