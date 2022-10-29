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
        class DexMethods;

        using methodid_t = std::unique_ptr<MethodID>;
        using methods_t = std::vector<methodid_t>;
        using dexmethods_t = std::unique_ptr<DexMethods>;

        class MethodID
        {
        public:
            MethodID(std::uint16_t class_idx,
                     std::uint16_t proto_idx,
                     std::uint32_t name_idx,
                     DexStrings *dex_strings,
                     DexTypes *dex_types,
                     DexProtos *dex_protos);

            ~MethodID() = default;

            Type *get_method_class()
            {
                return class_idx.second;
            }

            ProtoID *get_method_prototype()
            {
                return proto_idx.second;
            }

            std::string *get_method_name()
            {
                return name_idx.second;
            }

            std::string get_method_str();

            friend std::ostream &operator<<(std::ostream &os, const MethodID &entry);

        private:
            std::pair<std::uint16_t, Type *> class_idx;
            std::pair<std::uint16_t, ProtoID *> proto_idx;
            std::pair<std::uint32_t, std::string *> name_idx;
        };

        class DexMethods
        {
        public:
            DexMethods(std::ifstream &input_file,
                       std::uint32_t number_of_methods,
                       std::uint32_t offset,
                       DexStrings *dex_strings,
                       DexTypes *dex_types,
                       DexProtos *dex_protos);

            ~DexMethods() = default;

            std::uint32_t get_number_of_methods()
            {
                return number_of_methods;
            }

            const methods_t &get_method_ids() const
            {
                return method_ids;
            }

            MethodID* get_method_by_order(size_t pos);

            friend std::ostream &operator<<(std::ostream &os, const DexMethods &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexMethods &entry);

        private:
            bool parse_methods(std::ifstream &input_file);

            std::uint32_t number_of_methods;
            std::uint32_t offset;
            DexStrings *dex_strings;
            DexTypes *dex_types;
            DexProtos *dex_protos;

            methods_t method_ids;
        };
    }
}

#endif