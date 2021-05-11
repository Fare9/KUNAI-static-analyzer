/***
 * File: dex_encoded.hpp
 * Author: @Farenain
 * 
 * All the different Encoded types from
 * DEX format, I think is a good way
 * to separate them from other logic of
 * the parser.
 */

#ifndef DEX_ENCODED_HPP
#define DEX_ENCODED_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <map>

#include "dex_fields.hpp"
#include "dex_methods.hpp"
#include "dex_dvm_types.hpp"

namespace KUNAI {
    namespace DEX {

        class EncodedValue
        {
        public:
            EncodedValue(std::ifstream& input_file);
            ~EncodedValue();

            std::vector<std::uint8_t> get_values();
            std::vector<std::shared_ptr<EncodedValue>> get_array();

        private:
            std::vector<std::uint8_t> values;
            std::vector<std::shared_ptr<EncodedValue>> array;
        };

        class EncodedArray
        {
        public:
            EncodedArray(std::ifstream& input_file);
            ~EncodedArray();
        private:
            std::uint64_t size;
            std::vector<std::shared_ptr<EncodedValue>> values;
        };

        class EncodedArrayItem
        {
        public:
            EncodedArrayItem(std::ifstream& input_file);
            ~EncodedArrayItem();

            std::shared_ptr<EncodedArray> get_encoded_array();
        private:
            std::shared_ptr<EncodedArray> array;
        };

        class EncodedField
        {
        public:
            EncodedField(FieldID* field_idx, std::uint64_t access_flags);
            ~EncodedField();

            FieldID* get_field();
            DVMTypes::ACCESS_FLAGS get_access_flags();

        private:
            FieldID* field_idx;
            DVMTypes::ACCESS_FLAGS access_flags;
        };

        class EncodedMethod
        {
        public:
            EncodedMethod(MethodID* method_id, std::uint64_t access_flags, std::uint64_t code_off);
            ~EncodedMethod();

            MethodID* get_method();
            DVMTypes::ACCESS_FLAGS get_access_flags();
            std::uint64_t get_code_offset();

        private:
            MethodID* method_id;
            DVMTypes::ACCESS_FLAGS access_flags;
            std::uint64_t code_off;
        };

    }
}

#endif