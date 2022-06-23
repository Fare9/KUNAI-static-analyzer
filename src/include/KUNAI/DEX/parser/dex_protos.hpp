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

#ifndef DEX_PROTOS_HPP
#define DEX_PROTOS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <map>

#include "KUNAI/Exceptions/exceptions.hpp"
#include "KUNAI/Utils/utils.hpp"
#include "KUNAI/DEX/parser/dex_strings.hpp"
#include "KUNAI/DEX/parser/dex_types.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class ProtoID;

        using protoid_t = std::shared_ptr<ProtoID>;

        class ProtoID
        {
        public:
            ProtoID(std::uint32_t shorty_idx,
                    std::uint32_t return_type_idx,
                    std::uint32_t parameters_off,
                    std::ifstream &input_file,
                    dexstrings_t& dex_strings,
                    dextypes_t& dex_types);
            ~ProtoID() = default;

            size_t get_number_of_parameters()
            {
                return parameters.size();
            }

            std::vector<type_t>& get_parameters()
            {
                return parameters;
            }

            
            type_t get_parameter_type_by_order(size_t pos);


            type_t get_return_idx()
            {
                return return_type_idx;
            }

            std::string *get_shorty_idx()
            {
                return shorty_idx;
            }

            std::string get_proto_str();

        private:
            bool parse_parameters(std::ifstream &input_file,
                                  dexstrings_t& dex_strings,
                                  dextypes_t& dex_types);

            std::string *shorty_idx;
            type_t return_type_idx;
            std::uint32_t parameters_off;
            std::vector<type_t> parameters;
        };

        class DexProtos;

        using dexprotos_t = std::shared_ptr<DexProtos>;

        class DexProtos
        {
        public:
            DexProtos(std::ifstream &input_file,
                      std::uint64_t file_size,
                      std::uint32_t number_of_protos,
                      std::uint32_t offset,
                      dexstrings_t& dex_strings,
                      dextypes_t& dex_types);

            ~DexProtos() = default;

            std::uint32_t get_number_of_protos()
            {
                return number_of_protos;
            }

            protoid_t get_proto_by_order(size_t pos);

            const std::vector<protoid_t>& get_protos() const
            {
                return proto_ids;
            }

            friend std::ostream &operator<<(std::ostream &os, const DexProtos &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexProtos &entry);

        private:
            bool parse_protos(std::ifstream &input_file, std::uint64_t file_size);

            std::uint32_t number_of_protos;
            std::uint32_t offset;
            dexstrings_t& dex_strings;
            dextypes_t& dex_types;

            std::vector<protoid_t> proto_ids;
        };
    }
}

#endif
