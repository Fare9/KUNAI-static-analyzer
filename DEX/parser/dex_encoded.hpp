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

namespace KUNAI
{
    namespace DEX
    {

        class EncodedValue
        {
        public:
            EncodedValue(std::ifstream &input_file);
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
            EncodedArray(std::ifstream &input_file);
            ~EncodedArray();

        private:
            std::uint64_t size;
            std::vector<std::shared_ptr<EncodedValue>> values;
        };

        class EncodedArrayItem
        {
        public:
            EncodedArrayItem(std::ifstream &input_file);
            ~EncodedArrayItem();

            std::shared_ptr<EncodedArray> get_encoded_array();

        private:
            std::shared_ptr<EncodedArray> array;
        };

        class EncodedField
        {
        public:
            EncodedField(FieldID *field_idx, std::uint64_t access_flags);
            ~EncodedField();

            FieldID *get_field();
            DVMTypes::ACCESS_FLAGS get_access_flags();

        private:
            FieldID *field_idx;
            DVMTypes::ACCESS_FLAGS access_flags;
        };

        class EncodedTypePair
        {
        public:
            EncodedTypePair(std::uint64_t type_idx,
                            std::uint64_t addr,
                            std::shared_ptr<DexTypes> dex_types);
            ~EncodedTypePair();

            Type *get_exception_type();
            std::uint64_t get_exception_handler_addr();

        private:
            std::map<std::uint64_t, Type *> type_idx; // type of the exception to catch
            std::uint64_t addr;                       // bytecode address of associated exception handler
        };

        class EncodedCatchHandler
        {
        public:
            EncodedCatchHandler(std::ifstream &input_file,
                                std::uint64_t file_size,
                                std::shared_ptr<DexTypes> dex_types);
            ~EncodedCatchHandler();

            bool has_explicit_typed_catches();
            std::uint64_t get_size_of_handlers();
            std::shared_ptr<EncodedTypePair> get_handler_by_pos(std::uint64_t pos);
            std::uint64_t get_catch_all_addr();

        private:
            bool parse_encoded_type_pairs(std::ifstream &input_file,
                                          std::uint64_t file_size,
                                          std::shared_ptr<DexTypes> dex_types);

            std::int64_t encoded_type_pair_size;
            std::vector<std::shared_ptr<EncodedTypePair>> handlers;
            std::uint64_t catch_all_addr;
        };

        class TryItem
        {
        public:
            struct try_item_struct_t
            {
                std::uint16_t start_addr;  // start address of block of code covered by this entry.
                                           // Count of 16-bit code units to start of first.
                std::uint16_t insn_count;  // number of 16-bit code units covered by this entry.
                std::uint16_t handler_off; // offset in bytes from starts of associated encoded_catch_handler_list to
                                           // encoded_catch_handler for this entry.
            };

            TryItem(try_item_struct_t try_item_struct);
            ~TryItem();

        private:
            try_item_struct_t try_item_struct;
        };

        class CodeItemStruct
        {
        public:
            struct code_item_struct_t
            {
                std::uint16_t registers_size; // number of registers used in the code
                std::uint16_t ins_size;       // number of words of incoming arguments to the method
                std::uint16_t outs_size;      // number of words of outgoing argument space required by code for method invocation.
                std::uint16_t tries_size;     // number of try_items, it can be zero.
                std::uint32_t debug_info_off; // offset to debug_info_item
                std::uint32_t insns_size;     // size of instructions list, in 16-bit code units
            };

            CodeItemStruct(std::ifstream &input_file,
                           std::uint64_t file_size,
                           code_item_struct_t code_item,
                           std::shared_ptr<DexTypes> dex_types);
            ~CodeItemStruct();

            std::uint16_t get_number_of_registers_in_code();
            std::uint16_t get_number_of_incoming_arguments();
            std::uint16_t get_number_of_outgoing_arguments();
            std::uint16_t get_number_of_try_items();
            std::shared_ptr<TryItem> get_try_item_by_pos(std::uint64_t pos);
            std::uint16_t get_number_of_instructions();
            std::uint16_t get_instruction_by_pos(std::uint16_t pos);
            std::uint64_t get_encoded_catch_handler_list_size();
            std::shared_ptr<EncodedCatchHandler> get_encoded_catch_handler_by_pos(std::uint64_t pos);

        private:
            bool parse_code_item_struct(std::ifstream &input_file, std::uint64_t file_size, std::shared_ptr<DexTypes> dex_types);

            code_item_struct_t code_item;
            std::vector<std::uint16_t> instructions;
            std::vector<std::shared_ptr<TryItem>> try_items;
            std::vector<std::shared_ptr<EncodedCatchHandler>> encoded_catch_handler_list;
        };

        class EncodedMethod
        {
        public:
            EncodedMethod(MethodID *method_id,
                          std::uint64_t access_flags,
                          std::uint64_t code_off,
                          std::ifstream &input_file,
                          std::uint64_t file_size,
                          std::shared_ptr<DexTypes> dex_types);
            ~EncodedMethod();

            MethodID *get_method();
            DVMTypes::ACCESS_FLAGS get_access_flags();
            std::uint64_t get_code_offset();
            std::shared_ptr<CodeItemStruct> get_code_item();

        private:
            bool parse_code_item(std::ifstream &input_file, std::uint64_t file_size, std::shared_ptr<DexTypes> dex_types);

            MethodID *method_id;
            DVMTypes::ACCESS_FLAGS access_flags;
            std::uint64_t code_off;
            std::shared_ptr<CodeItemStruct> code_item;
        };

    }
}

#endif