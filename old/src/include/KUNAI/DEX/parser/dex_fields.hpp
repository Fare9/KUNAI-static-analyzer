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
#include <utility>
#include <optional>

#include "KUNAI/Exceptions/exceptions.hpp"
#include "KUNAI/Utils/utils.hpp"
#include "KUNAI/DEX/parser/dex_strings.hpp"
#include "KUNAI/DEX/parser/dex_types.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class FieldID;
        class DexFields;

        using fieldid_t = std::unique_ptr<FieldID>;
        using fields_t = std::vector<fieldid_t>;

        using dexfields_t = std::unique_ptr<DexFields>;

        class FieldID
        {
        public:
            /**
             * @brief Construct a new Field of Android, this contains
             *        the class, the type and the name.
             *
             * @param class_idx
             * @param type_idx
             * @param name_idx
             * @param dex_strings
             * @param dex_types
             */
            FieldID(std::uint16_t class_idx,
                    std::uint16_t type_idx,
                    std::uint32_t name_idx,
                    DexStrings *dex_strings,
                    DexTypes *dex_types);

            ~FieldID() = default;

            /**
             * @brief Get the class from the field
             *
             * @return Type*
             */
            Type *get_class_idx();

            /**
             * @brief Get the type from the field
             *
             * @return Type*
             */
            Type *get_type_idx();

            /**
             * @brief Get the name from the field
             *
             * @return std::string*
             */
            std::string *get_name_idx();

            /**
             * @brief Get a string representation from the field.
             *
             * @return std::string
             */
            std::string get_field_str();

            friend std::ostream &operator<<(std::ostream &os, const FieldID &entry);

        private:
            std::pair<std::uint16_t, Type *> class_idx;
            std::pair<std::uint16_t, Type *> type_idx;
            std::pair<std::uint32_t, std::string *> name_idx;
        };

        class DexFields
        {
        public:
            /**
             * @brief Construct a new Dex Fields object, this is
             *        the representation of the Android Fields,
             *        it contains a list of fields.
             *
             * @param input_file
             * @param number_of_fields
             * @param offset
             * @param dex_strings
             * @param dex_types
             */
            DexFields(std::ifstream &input_file,
                      std::uint32_t number_of_fields,
                      std::uint32_t offset,
                      DexStrings *dex_strings,
                      DexTypes *dex_types);

            ~DexFields() = default;

            /**
             * @brief Get the number of fields object
             *
             * @return std::uint64_t
             */
            std::uint64_t get_number_of_fields()
            {
                return number_of_fields;
            }

            /**
             * @brief Get the fields object
             *
             * @return const fields_t&
             */
            const fields_t &get_fields() const
            {
                return field_ids;
            }

            FieldID *get_field_id_by_order(size_t pos);

            friend std::ostream &operator<<(std::ostream &os, const DexFields &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexFields &entry);

        private:
            bool parse_fields(std::ifstream &input_file);

            std::uint32_t number_of_fields;
            std::uint32_t offset;
            DexStrings *dex_strings;
            DexTypes *dex_types;

            fields_t field_ids;
        };

    }
}

#endif