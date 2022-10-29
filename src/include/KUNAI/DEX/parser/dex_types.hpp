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

#include "KUNAI/Exceptions/exceptions.hpp"
#include "KUNAI/Utils/utils.hpp"
#include "KUNAI/DEX/parser/dex_strings.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class Type;
        class Fundamental;
        class Class;
        class Array;
        class Unknown;
        
        class DexTypes;

        
        using type_t        = std::unique_ptr<Type>;
        using fundamental_t = std::unique_ptr<Fundamental>;
        using class_t       = std::unique_ptr<Class>;
        using array_t       = std::unique_ptr<Array>;
        using unknown_t     = std::unique_ptr<Unknown>;

        using dextypes_t = std::unique_ptr<DexTypes>;

        class Type
        {
        public:
            enum type_e
            {
                FUNDAMENTAL,
                CLASS,
                ARRAY,
                UNKNOWN
            };

            /**
             * @brief Constructor of Type base class.
             * @param type: Type of object from Type::type_e enum.
             * @param raw: raw string with the name.
             */
            Type(type_e type, std::string raw);

            /**
             * @brief Type destructor.
             */
            virtual ~Type() = default;

            virtual type_e get_type() = 0;
            virtual std::string print_type() = 0;

            /**
             * @brief get raw string from the type object.
             * @return raw: string from the type.
             */
            std::string& get_raw()
            {
                return raw;
            }

        private:
            enum type_e type;
            std::string raw;
        };

        class Fundamental : public Type
        {
        public:
            enum fundamental_e
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
             * @param f_type: Fundamental::fundamental_e enum value, type of fundamental.
             * @param name: std::string with name of fundamental type.
             */
            Fundamental(fundamental_e f_type,
                        std::string name);

            /**
             * @brief Fundamental Destructor
             */
            ~Fundamental() = default;

            /**
             * @brief return Type::type_e of object, in this case FUNDAMENTAL.
             * @return Type::type_e FUNDAMENTAL.
             */
            type_e get_type()
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
             * @brief Get fundamental type from enum Fundamental::fundamental_e.
             * @return Fundamental::fundamental_e value.
             */
            fundamental_e get_fundamental_type()
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
            enum fundamental_e f_type;
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
             * @brief get Type::type_e enum value for Class object.
             * @return Type::type_e with value CLASS.
             */
            type_e get_type()
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
            std::string& get_name()
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
            Array(std::vector<type_t> array, std::string raw);

            /**
             * @brief Destructor of Array class.
             */
            ~Array() = default;

            /**
             * @brief get type from Type::type_e enum.
             * @return return ARRAY from Type::type_e.
             */
            type_e get_type()
            {
                return ARRAY;
            }

            /**
             * @brief get type from Type::type_e enum as string.
             * @return ARRAY from Type::type_e as string.
             */
            std::string print_type()
            {
                return "Array";
            }

            /**
             * @brief get the vector of the array of types.
             * @return std::vector<type_t> with all the types in the array.
             */
            const std::vector<type_t> &get_array() const
            {
                return array;
            }

        private:
            std::vector<type_t> array;
        };

        class Unknown : public Type
        {
        public:
            /**
             * @brief Constructor of unrecognized type.
             * @param type: for parent class.
             * @param raw: for parent class.
             */
            Unknown(type_e type, std::string raw);

            /**
             * @brief Destructor of unrecognized type.
             */
            ~Unknown() = default;

            /**
             * @brief return Type::type_e of UNKNOWN.
             * @return Type::type_e UNKNOWN.
             */
            type_e get_type()
            {
                return UNKNOWN;
            }

            /**
             * @brief return Type::type_e of UNKNOWN as std::string.
             * @return Type::type_e UNKNOWN as std::string.
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
                     DexStrings* dex_str);

            /**
             * @brief Destructor of DexTypes class, clear all the Types.
             */
            ~DexTypes();

            /**
             * @brief Get a type by a given type id as it appears in DEX.
             * @param type_id: type to retrieve by its id.
             * @return Type* object.
             */
            Type* get_type_by_id(std::uint32_t type_id);

            /**
             * @brief Get a type by a given position from the map.
             * @param pos: position of the Type in the map.
             * @return Type* object.
             */
            Type* get_type_from_order(std::uint32_t pos);

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
             * @return type_t object created.
             */
            type_t parse_type(std::string name);

            /**
             * @brief parser method for DEX types. It will make use of method parse_type.
             * @param input_file: file where to read the types.
             * @return true if everything was okay, false in other way.
             */
            bool parse_types(std::ifstream &input_file);

            // variables from types
            std::map<std::uint32_t, type_t> types;
            std::uint32_t number_of_types;
            std::uint32_t offset;
            DexStrings* dex_str;
        };
    }
}

#endif