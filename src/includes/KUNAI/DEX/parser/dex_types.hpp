/**
 * @file dex_types.hpp
 * @author @Farenain
 *
 * @brief Serie of types which points to string values.
 *
 *
 * we will get the string values too.
 * Each one of the types point to an index in the
 * string lookup table.
 *
 * type_id_item[]
 *
 * type_id_item:
 *  descriptor_idx: uint -> index into string_ids list
 *
 * types:
 *  X --> strings[X] --> class1
 *  Y --> strings[Y] --> class2
 *  Z --> strings[Z] --> class3
 *  ...
 */

#ifndef DEX_TYPES_HPP
#define DEX_TYPES_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <memory>

#include "exceptions.hpp"
#include "utils.hpp"
#include "dex_strings.hpp"

namespace KUNAI
{
    namespace DEX
    {

        class Type
        {
        public:
            enum type_t
            {
                FUNDAMENTAL,
                CLASS,
                ARRAY,
                UNKNOWN
            };

            /**
             * @brief Constructor of Type base class.
             * @param type: Type of object from Type::type_t enum.
             * @param raw: raw string with the name.
             */
            Type(type_t type, std::string raw);

            /**
             * @brief Type destructor.
             */
            virtual ~Type() = default;

            virtual type_t get_type() = 0;
            virtual std::string print_type() = 0;

            /**
             * @brief get raw string from the type object.
             * @return raw: string from the type.
             */
            std::string get_raw()
            {
                return this->raw;
            }

        private:
            enum type_t type;
            std::string raw;
        };

        class Fundamental : public Type
        {
        public:
            enum fundamental_t
            {
                BOOLEAN,
                BYTE,
                CHAR,
                DOUBLE,
                FLOAT,
                INT,
                LONG,
                SHORT,
                VOID
            };

            /**
             * @brief Constructor of derived Fundamental class.
             * @param f_type: Fundamental::fundamental_t enum value, type of fundamental.
             * @param name: std::string with name of fundamental type.
             */
            Fundamental(fundamental_t f_type,
                        std::string name);

            /**
             * @brief Fundamental Destructor
             */
            ~Fundamental() = default;

            /**
             * @brief return Type::type_t of object, in this case FUNDAMENTAL.
             * @return Type::type_t FUNDAMENTAL.
             */
            type_t get_type()
            {
                return FUNDAMENTAL;
            }

            /**
             * @brief return std::string with the FUndamental type.
             * @return std::string "Fundamental"
             */
            std::string print_type()
            {
                return "Fundamental";
            }

            /**
             * @brief Get fundamental type from enum Fundamental::fundamental_t.
             * @return Fundamental::fundamental_t value.
             */
            fundamental_t get_fundamental_type()
            {
                return f_type;
            }

            /**
             * @brief return the type of Fundamental of the object.
             * @return std::string with string representation of Fundamental::fundamental_t value.
             */
            std::string print_fundamental_type();

            /**
             * @brief get name of fundamental type.
             * @return std::string with name of fundamental type.
             */
            std::string get_name()
            {
                return name;
            }

        private:
            enum fundamental_t f_type;
            std::string name;
        };

        class Class : public Type
        {
        public:
            /**
             * @brief Constructor of derived Class class.
             * @param name: std::string with name of Class type.
             */
            Class(std::string name);

            /**
             * @brief Destructor of derived Class class.
             */
            ~Class() = default;

            /**
             * @brief get Type::type_t enum value for Class object.
             * @return Type::type_t with value CLASS.
             */
            type_t get_type()
            {
                return CLASS;
            }

            /**
             * @brief get Type::type_t enum value as string.
             * @return Type::type_t as string.
             */
            std::string print_type()
            {
                return "Class";
            }

            /**
             * @brief get name of Class type.
             * @return std::string with name of Class type.
             */
            std::string get_name()
            {
                return name;
            }

        private:
            std::string name;
        };

        class Array : public Type
        {
        public:
            /**
             * @brief Constructor of derived Array class.
             * @param array: a std::vector with types inside of the array.
             * @param name: std::string with name of Array type.
             */
            Array(std::vector<Type *> array, std::string raw);

            /**
             * @brief Destructor of Array class, destroy all the values inside of the array.
             */
            ~Array();

            /**
             * @brief get type from Type::type_t enum.
             * @return return ARRAY from Type::type_t.
             */
            type_t get_type()
            {
                return ARRAY;
            }

            /**
             * @brief get type from Type::type_t enum as string.
             * @return ARRAY from Type::type_t as string.
             */
            std::string print_type()
            {
                return "Array";
            }

            /**
             * @brief get the vector of the array of types.
             * @return std::vector<Type *> with all the types in the array.
             */
            const std::vector<Type *> &get_array() const
            {
                return array;
            }

        private:
            std::vector<Type *> array;
        };

        class Unknown : public Type
        {
        public:
            /**
             * @brief Constructor of unrecognized type.
             * @param type: for parent class.
             * @param raw: for parent class.
             */
            Unknown(type_t type, std::string raw);

            /**
             * @brief Destructor of unrecognized type.
             */
            ~Unknown() = default;

            /**
             * @brief return Type::type_t of UNKNOWN.
             * @return Type::type_t UNKNOWN.
             */
            type_t get_type()
            {
                return UNKNOWN;
            }

            /**
             * @brief return Type::type_t of UNKNOWN as std::string.
             * @return Type::type_t UNKNOWN as std::string.
             */
            std::string print_type()
            {
                return "Unknown";
            }
        };

        class DexTypes
        {
        public:
            /**
             * @brief Constructor of DexTypes class used to parse types from DEX.
             * @param input_file: file where to read the types.
             * @param number_of_types: number of types to parse.
             * @param types_offsets: offset from the file where the types are.
             * @param dex_str: strings from the DEX.
             */
            DexTypes(std::ifstream &input_file,
                     std::uint32_t number_of_types,
                     std::uint32_t types_offsets,
                     std::shared_ptr<DexStrings> dex_str);

            /**
             * @brief Destructor of DexTypes class, clear all the Types.
             */
            ~DexTypes();

            /**
             * @brief Get a type by a given type id as it appears in DEX.
             * @param type_id: type to retrieve by its id.
             * @return Type* object.
             */
            Type *get_type_by_id(std::uint32_t type_id);

            /**
             * @brief Get a type by a given position from the map.
             * @param pos: position of the Type in the map.
             * @return Type* object.
             */
            Type *get_type_from_order(std::uint32_t pos);

            /**
             * @brief Return the number of types in map.
             * @return number of type objects.
             */
            std::uint32_t get_number_of_types()
            {
                return number_of_types;
            }

            /**
             * @brief Return the offset where Types start.
             * @return offset where Types are in DEX file.
             */
            std::uint32_t get_offset()
            {
                return offset;
            }

            friend std::ostream &operator<<(std::ostream &os, const DexTypes &entry);
            friend std::fstream &operator<<(std::fstream &fos, const DexTypes &entry);

        private:
            // private methods

            /**
             * @brief Method to create a Type given a raw name.
             * @param name: raw name of the type.
             * @return Type* object created.
             */
            Type *parse_type(std::string name);

            /**
             * @brief parser method for DEX types. It will make use of method parse_type.
             * @param input_file: file where to read the types.
             * @return true if everything was okay, false in other way.
             */
            bool parse_types(std::ifstream &input_file);

            // variables from types
            std::map<std::uint32_t, Type *> types;
            std::uint32_t number_of_types;
            std::uint32_t offset;
            std::shared_ptr<DexStrings> dex_str;
        };
    }
}

#endif