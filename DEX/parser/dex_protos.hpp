/***
 * @file dex_protos.hpp
 * @author @Farenain
 * 
 * @brief Method prototypes in the java files of Android.
 *        Using proto_ids with relevant type_ids, the method_ids
 *        are assembled.
 * 
 * Each proto structure is:
 *  ProtoID {
 *      uint shorty_idx, # OFFSET to a string with return type
 *      uint return_type_idx, # OFFSET to a type which points to string with return type
 *      uint parameters_off # ---------| Offset to parameters in type list type
 *  }                                  |
 *                                     |
 *  ------------------------------------
 *  |
 *  v
 *  type_list {
 *      uint size, # Size of the list in entries
 *      list type_item[size] # ---------|
 *  }                                   |
 *                                      |
 *  -------------------------------------
 *  |
 *  v
 *  type_item {
 *      ushort type_idx # type_idx of each member
 *  }
 * 
 *  ProtoID[] protos
 */

#pragma once

#ifndef DEX_PROTOS_HPP
#define DEX_PROTOS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <map>

#include "exceptions.hpp"
#include "utils.hpp"
#include "dex_strings.hpp"
#include "dex_types.hpp"

namespace KUNAI {
    namespace DEX {

        class ProtoID
        {
        public:
            ProtoID(std::uint32_t shorty_idx, 
                    std::uint32_t return_type_idx, 
                    std::uint32_t parameters_off,
                    std::ifstream& input_file,
                    std::shared_ptr<DexStrings> dex_strings,
                    std::shared_ptr<DexTypes> dex_types);
            ~ProtoID();

            size_t get_number_of_parameters();
            Type* get_parameter_type_by_order(size_t pos);
            Type* get_return_idx();
            std::string* get_shorty_idx();

        private:
            bool parse_parameters(std::ifstream& input_file, 
                                    std::shared_ptr<DexStrings> dex_strings,
                                    std::shared_ptr<DexTypes> dex_types);
            std::map<std::uint32_t, std::string*> shorty_idx;
            std::map<std::uint32_t, Type*> return_type_idx;
            std::uint32_t parameters_off;
            std::map<std::uint16_t, Type*> parameters;
        };

        class DexProtos
        {
        public:
            DexProtos(std::ifstream& input_file,
                        std::uint64_t file_size,
                        std::uint32_t number_of_protos,
                        std::uint32_t offset,
                        std::shared_ptr<DexStrings> dex_strings,
                        std::shared_ptr<DexTypes> dex_types);
            
            ~DexProtos();

            std::uint32_t get_number_of_protos();
            ProtoID* get_proto_by_order(size_t pos);

            friend std::ostream& operator<<(std::ostream& os, const DexProtos& entry);
            friend std::fstream& operator<<(std::fstream& fos, const DexProtos& entry);
        private:
            bool parse_protos(std::ifstream& input_file, std::uint64_t file_size);

            std::uint32_t number_of_protos;
            std::uint32_t offset;
            std::shared_ptr<DexStrings> dex_strings;
            std::shared_ptr<DexTypes> dex_types;

            std::vector<ProtoID*> proto_ids;
        };
    }
}

#endif
