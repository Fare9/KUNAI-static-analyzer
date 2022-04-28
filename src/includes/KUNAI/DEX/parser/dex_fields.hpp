/***
 * @file dex_fields.hpp
 * @author @Farenain
 *
 * @brief Android fields of the Java code, composed by class name,
 *        type and name of the field.
 *
 *  FieldID {
 *      ushort class_idx, # type_id for class name
 *      ushort type_idx, # type_id of field type
 *      uint name_idx # Field name
 *  }
 *
 *  FieldID[] dex_fields;
 */

#ifndef DEX_FIELDS_HPP
#define DEX_FIELDS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <map>

#include "exceptions.hpp"
#include "utils.hpp"
#include "dex_strings.hpp"
#include "dex_types.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class FieldID;

        using fieldid_t = std::shared_ptr<FieldID>;

        class FieldID
        {
        public:
            FieldID(std::uint16_t class_idx,
                    std::uint16_t type_idx,
                    std::uint32_t name_idx,
                    dexstrings_t& dex_strings,
                    dextypes_t& dex_types);
            ~FieldID() = default;

            type_t get_class_idx();
            type_t get_type_idx();
            std::string *get_name_idx();

            friend std::ostream &operator<<(std::ostream &os, const FieldID &entry);

        private:
            std::map<std::uint16_t, type_t> class_idx;
            std::map<std::uint16_t, type_t> type_idx;
            std::map<std::uint32_t, std::string *> name_idx;
        };

        class DexFields;

        using dexfields_t = std::shared_ptr<DexFields>;

        class DexFields
        {
        public:
            DexFields(std::ifstream &input_file,
                      std::uint32_t number_of_fields,
                      std::uint32_t offset,
                      dexstrings_t& dex_strings,
                      dextypes_t& dex_types);

            ~DexFields() = default;

            std::uint64_t get_number_of_fields()
            {
                return number_of_fields;
            }

            std::vector<fieldid_t>& get_fields()
            {
                return field_ids;
            }

            fieldid_t get_field_id_by_order(size_t pos);

            friend std::ostream &operator<<(std::ostream &os, const DexFields &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexFields &entry);

        private:
            bool parse_fields(std::ifstream &input_file);

            std::uint32_t number_of_fields;
            std::uint32_t offset;
            dexstrings_t& dex_strings;
            dextypes_t& dex_types;

            std::vector<fieldid_t> field_ids;
        };

    }
}

#endif