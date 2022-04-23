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
#include <map>

#include "exceptions.hpp"
#include "utils.hpp"
#include "dex_strings.hpp"
#include "dex_types.hpp"
#include "dex_protos.hpp"

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

            Type *get_method_class()
            {
                return class_idx.begin()->second;
            }

            ProtoID *get_method_prototype()
            {
                return proto_idx.begin()->second;
            }

            std::string *get_method_name()
            {
                return name_idx.begin()->second;
            }

            friend std::ostream &operator<<(std::ostream &os, const MethodID &entry);

        private:
            std::map<std::uint16_t, Type *> class_idx;
            std::map<std::uint16_t, ProtoID *> proto_idx;
            std::map<std::uint32_t, std::string *> name_idx;
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
            ~DexMethods();

            std::uint32_t get_number_of_methods()
            {
                return number_of_methods;
            }

            MethodID *get_method_by_order(size_t pos);

            friend std::ostream &operator<<(std::ostream &os, const DexMethods &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexMethods &entry);

        private:
            bool parse_methods(std::ifstream &input_file);

            std::uint32_t number_of_methods;
            std::uint32_t offset;
            dexstrings_t& dex_strings;
            dextypes_t& dex_types;
            dexprotos_t& dex_protos;

            std::vector<MethodID *> method_ids;
        };
    }
}

#endif