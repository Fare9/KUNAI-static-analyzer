/***
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

#pragma once

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

namespace KUNAI {
    namespace DEX {

        class Type
        {
        public:
            enum type_t {
                FUNDAMENTAL,
                CLASS,
                ARRAY,
                UNKNOWN
            };
        
        Type(type_t type, std::string raw);
        virtual ~Type();

        virtual type_t get_type() = 0;
        virtual std::string print_type() = 0;

        std::string get_raw();

        private:
            enum type_t type;
            std::string raw;
        };

        class Fundamental : public Type
        {
        public:
            enum fundamental_t {
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
            Fundamental(fundamental_t f_type, 
                        std::string name);
            ~Fundamental();

            type_t get_type();
            std::string print_type();

            fundamental_t get_fundamental_type();
            std::string print_fundamental_type();

            std::string get_name();
        private:
            enum fundamental_t f_type;
            std::string name;

            
        };

        class Class : public Type
        {
        public:
            Class(std::string name);
            ~Class();

            type_t get_type();
            std::string print_type();

            std::string get_name();

            bool is_external_class();
        private:
            std::string name;

            std::vector<std::string> external_classes {
                "Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/", "Ljavax/", "Lorg/apache/",
                          "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/", "Ljunit/"
            };
        };

        class Array : public Type
        {
        public:
            Array(std::vector<Type*> array, std::string raw);
            ~Array();

            type_t get_type();
            std::string print_type();

            std::vector<Type*> get_array();
        private:
            std::vector<Type*> array;
        };

        class Unknown : public Type
        {
        public:
            Unknown(type_t type, std::string raw);
            ~Unknown();

            type_t get_type();
            std::string print_type();
        };



        class DexTypes
        {
        public:
            DexTypes(std::ifstream& input_file, 
                std::uint32_t number_of_types, 
                std::uint32_t types_offsets,
                std::shared_ptr<DexStrings> dex_str);
            ~DexTypes();

            Type* get_type_by_id(std::uint32_t type_id);
            Type* get_type_from_order(std::uint32_t pos);

            std::uint32_t get_number_of_types();
            std::uint32_t get_offset();

            friend std::ostream& operator<<(std::ostream& os, const DexTypes& entry);
            friend std::fstream& operator<<(std::fstream& fos, const DexTypes& entry);
        private:
            // private methods
            Type* parse_type(std::string name);
            bool parse_types(std::ifstream& input_file);
            
            // variables from types
            std::map<std::uint32_t, Type*> types;
            std::uint32_t number_of_types;
            std::uint32_t offset;
            std::shared_ptr<DexStrings> dex_str;
        };
    }
}

#endif