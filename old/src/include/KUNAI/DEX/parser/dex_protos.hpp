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
        class DexProtos;

        using protoid_t = std::unique_ptr<ProtoID>;
        using protos_t = std::vector<protoid_t>;

        using dexprotos_t = std::unique_ptr<DexProtos>;

        class ProtoID
        {
        public:
            /**
             * @brief Construct a new Proto ID object, this represent
             *        a prototype of an Android method.
             *
             * @param shorty_idx
             * @param return_type_idx
             * @param parameters_off
             * @param input_file
             * @param dex_strings
             * @param dex_types
             */
            ProtoID(std::uint32_t shorty_idx,
                    std::uint32_t return_type_idx,
                    std::uint32_t parameters_off,
                    std::ifstream &input_file,
                    DexStrings *dex_strings,
                    DexTypes *dex_types);

            ~ProtoID() = default;

            /**
             * @brief Get the number of parameters
             *
             * @return size_t
             */
            size_t get_number_of_parameters()
            {
                return parameters.size();
            }

            /**
             * @brief Get the parameters
             *
             * @return const std::vector<Type *>
             */
            const std::vector<Type *> get_parameters() const
            {
                return parameters;
            }

            /**
             * @brief Get a parameter by its position
             *
             * @param pos
             * @return Type*
             */
            Type *get_parameter_type_by_order(size_t pos);

            /**
             * @brief Get the returned type
             *
             * @return Type*
             */
            Type *get_return_idx()
            {
                return return_type_idx;
            }

            /**
             * @brief Get the proto as a string.
             *
             * @return std::string*
             */
            std::string *get_shorty_idx()
            {
                return shorty_idx;
            }

            /**
             * @brief Get prototype string correctly represented.
             *
             * @return std::string
             */
            std::string get_proto_str();

        private:
            bool parse_parameters(std::ifstream &input_file,
                                  DexStrings *dex_strings,
                                  DexTypes *dex_types);

            std::string *shorty_idx;
            Type *return_type_idx;
            std::uint32_t parameters_off;
            std::vector<Type *> parameters;
        };

        class DexProtos
        {
        public:
            DexProtos(std::ifstream &input_file,
                      std::uint64_t file_size,
                      std::uint32_t number_of_protos,
                      std::uint32_t offset,
                      DexStrings *dex_strings,
                      DexTypes *dex_types);

            ~DexProtos() = default;

            std::uint32_t get_number_of_protos()
            {
                return number_of_protos;
            }

            ProtoID* get_proto_by_order(size_t pos);

            const protos_t &get_protos() const
            {
                return proto_ids;
            }

            friend std::ostream &operator<<(std::ostream &os, const DexProtos &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexProtos &entry);

        private:
            bool parse_protos(std::ifstream &input_file, std::uint64_t file_size);

            std::uint32_t number_of_protos;
            std::uint32_t offset;
            DexStrings *dex_strings;
            DexTypes *dex_types;

            protos_t proto_ids;
        };
    }
}

#endif
