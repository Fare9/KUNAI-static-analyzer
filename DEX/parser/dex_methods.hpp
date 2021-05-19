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

#pragma once

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

namespace KUNAI {
    namespace DEX {
        class MethodID
        {
        public:
            MethodID(std::uint16_t class_idx,
                     std::uint16_t proto_idx,
                     std::uint32_t name_idx,
                     std::shared_ptr<DexStrings> dex_strings,
                     std::shared_ptr<DexTypes> dex_types,
                     std::shared_ptr<DexProtos> dex_protos);
            
            ~MethodID();

            Type* get_method_class();
            ProtoID* get_method_prototype();
            std::string* get_method_name();

            friend std::ostream& operator<<(std::ostream& os, const MethodID& entry);
        private:
            std::map<std::uint16_t, Type*> class_idx;
            std::map<std::uint16_t, ProtoID*> proto_idx;
            std::map<std::uint32_t, std::string*> name_idx;
        };

        class DexMethods
        {
        public:
            DexMethods(std::ifstream& input_file,
                        std::uint32_t number_of_methods,
                        std::uint32_t offset,
                        std::shared_ptr<DexStrings> dex_strings,
                        std::shared_ptr<DexTypes> dex_types,
                        std::shared_ptr<DexProtos> dex_protos);
            ~DexMethods();

            std::uint32_t get_number_of_methods();
            MethodID* get_method_by_order(size_t pos);

            friend std::ostream& operator<<(std::ostream& os, const DexMethods& entry);
            friend std::fstream& operator<<(std::fstream& fos, const DexMethods& entry);
        private:
            bool parse_methods(std::ifstream& input_file);

            std::uint32_t number_of_methods;
            std::uint32_t offset;
            std::shared_ptr<DexStrings> dex_strings;
            std::shared_ptr<DexTypes> dex_types;
            std::shared_ptr<DexProtos> dex_protos;

            std::vector<MethodID*> method_ids;
        };
    }
}

#endif