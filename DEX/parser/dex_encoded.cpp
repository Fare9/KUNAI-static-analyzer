#include "dex_encoded.hpp"

namespace KUNAI {
    namespace DEX {
        EncodedValue::EncodedValue(std::ifstream& input_file)
        {
            std::uint8_t value, value_type, value_args;

            if (!KUNAI::read_data_file<std::uint8_t>(value, sizeof(std::uint8_t), input_file))
                throw exceptions::ParserReadingException("Error reading EncodedValue 'value' from file");
            
            value_type = value & 0x1f;
            value_args = ((value & 0xe0) >> 5);

            switch(value_type)
            {
            case DVMTypes::VALUE_BYTE:
            case DVMTypes::VALUE_SHORT:
            case DVMTypes::VALUE_CHAR:
            case DVMTypes::VALUE_INT:
            case DVMTypes::VALUE_LONG:
            case DVMTypes::VALUE_FLOAT:
            case DVMTypes::VALUE_DOUBLE:
            case DVMTypes::VALUE_STRING:
            case DVMTypes::VALUE_TYPE:
            case DVMTypes::VALUE_FIELD:
            case DVMTypes::VALUE_METHOD:
            case DVMTypes::VALUE_ENUM:
            {
                std::uint8_t aux;
                for (size_t i = 0; i < value_args; i++)
                {
                    if (!KUNAI::read_data_file<std::uint8_t>(aux, sizeof(std::uint8_t), input_file))
                        throw exceptions::ParserReadingException("Error reading EncodedValue 'aux' from file");
                    values.push_back(aux);
                }
                break;
            }
            case DVMTypes::VALUE_ARRAY:
            {
                std::uint64_t size = KUNAI::read_uleb128(input_file);
                for (size_t i = 0; i < size; i++)
                {
                    array.push_back(std::make_shared<EncodedValue>(input_file));
                }
                break;
            }
            case DVMTypes::VALUE_ANNOTATION:
            {
                // ToDo
                break;
            }
            case DVMTypes::VALUE_NULL:
            case DVMTypes::VALUE_BOOLEAN:
                break;
            }
        }
        
        EncodedValue::~EncodedValue()
        {
            if (!values.empty())
                values.clear();
        }

        std::vector<std::uint8_t> EncodedValue::get_values()
        {
            return values;
        }

        std::vector<std::shared_ptr<EncodedValue>> EncodedValue::get_array()
        {
            return array;
        }
        /***
         * EncodedArray
         */
        EncodedArray::EncodedArray(std::ifstream& input_file)
        {
            size = KUNAI::read_uleb128(input_file);
            std::shared_ptr<EncodedValue> encoded_value;

            for (size_t i = 0; i < size; i++)
            {
                encoded_value = std::make_shared<EncodedValue>(input_file);
                values.push_back(encoded_value);
            }
        }

        EncodedArray::~EncodedArray()
        {
            if (!values.empty())
                values.clear();
        }


        /***
         * EncodedArrayItem
         */
        EncodedArrayItem::EncodedArrayItem(std::ifstream& input_file)
        {
            this->array = std::make_shared<EncodedArray>(input_file);
        }

        EncodedArrayItem::~EncodedArrayItem() {}

        std::shared_ptr<EncodedArray> EncodedArrayItem::get_encoded_array()
        {
            return array;
        }
        /***
         * EncodedField
         */
        EncodedField::EncodedField(FieldID *field_idx, std::uint64_t access_flags)
        {
            this->field_idx = field_idx;
            this->access_flags = static_cast<DVMTypes::ACCESS_FLAGS>(access_flags);
        }

        EncodedField::~EncodedField() {}

        FieldID *EncodedField::get_field()
        {
            return field_idx;
        }

        DVMTypes::ACCESS_FLAGS EncodedField::get_access_flags()
        {
            return access_flags;
        }


        /***
         * EncodedTypePair
         * 
         * ToDo
         */
        EncodedTypePair::EncodedTypePair() {}
        EncodedTypePair::~EncodedTypePair() {}

        /***
         * EncodedCatchHandler
         * 
         * ToDo
         */
        EncodedCatchHandler::EncodedCatchHandler() {}
        EncodedCatchHandler::~EncodedCatchHandler() {}

        /***
         * TryItem
         * 
         * ToDo
         */
        TryItem::TryItem(try_item_struct_t try_item_struct)
        {
            this->try_item_struct = try_item_struct;
        }

        TryItem::~TryItem() {}

        /***
         * CodeItemStruct
         */
        CodeItemStruct::CodeItemStruct(std::ifstream& input_file, 
                                        std::uint64_t file_size, 
                                        code_item_struct_t code_item,
                                        std::shared_ptr<DexTypes> dex_types)
        {
            this->code_item = code_item;

            if (!parse_code_item_struct(input_file, file_size, dex_types))
                throw exceptions::ParserReadingException("Error reading CodeItemStruct");
        }

        CodeItemStruct::~CodeItemStruct() {}

        std::uint16_t CodeItemStruct::get_number_of_registers_in_code()
        {
            return code_item.registers_size;
        }

        std::uint16_t CodeItemStruct::get_number_of_incoming_arguments()
        {
            return code_item.ins_size;
        }

        std::uint16_t CodeItemStruct::get_number_of_outgoing_arguments()
        {
            return code_item.outs_size;
        }

        std::uint16_t CodeItemStruct::get_number_of_try_items()
        {
            return code_item.tries_size;
        }
        
        std::shared_ptr<TryItem> CodeItemStruct::get_try_item_by_pos(std::uint64_t pos)
        {
            if (pos >= try_items.size())
                return nullptr;
            return try_items[pos];
        }

        std::uint16_t CodeItemStruct::get_number_of_instructions()
        {
            return code_item.insns_size;
        }

        std::uint64_t CodeItemStruct::get_encoded_catch_handler_list_size()
        {
            return encoded_catch_handler_list.size();
        }

        std::shared_ptr<EncodedCatchHandler> CodeItemStruct::get_encoded_catch_handler_by_pos(std::uint64_t pos)
        {
            if (pos >= encoded_catch_handler_list.size())
                return nullptr;
            return encoded_catch_handler_list[pos];
        }

        // ToDo
        bool CodeItemStruct::parse_code_item_struct(std::ifstream& input_file, 
                                                    std::uint64_t file_size,
                                                    std::shared_ptr<DexTypes> dex_types)
        {
            auto current_offset = input_file.tellg();

            size_t i;
            std::uint16_t instruction;
            TryItem::try_item_struct_t try_item_struct;
            std::shared_ptr<TryItem> try_item;
            std::uint64_t encoded_catch_handler_list_size;

            for (i = 0; i < code_item.insns_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint16_t>(instruction, sizeof(std::uint16_t), input_file))
                    return false;
                instructions.push_back(instruction);
            }

            if ((code_item.tries_size > 0) && (code_item.insns_size % 2 != 0))
            {
                // padding is only present in case tries_size is greater than 0,
                // and instructions size is odd.
                if (!KUNAI::read_data_file<std::uint16_t>(instruction, sizeof(std::uint16_t), input_file))
                    return false;
            }

            if (code_item.tries_size > 0)
            {
                for (i = 0; i < code_item.tries_size; i++)
                {
                    if (!KUNAI::read_data_file<TryItem::try_item_struct_t>(try_item_struct, sizeof(TryItem::try_item_struct_t), input_file))
                        return false;
                    
                    if (try_item_struct.handler_off > file_size)
                        throw exceptions::OutOfBoundException("Error reading try_item_struct_t.handler_off out of file bound");

                    try_item = std::make_shared<TryItem>(try_item_struct);
                    try_items.push_back(try_item);
                }

                encoded_catch_handler_list_size = KUNAI::read_uleb128(input_file);
            }

            input_file.seekg(current_offset);
            return true;
        }

        /***
         * EncodedMethod
         */
        EncodedMethod::EncodedMethod(MethodID *method_id,
                                     std::uint64_t access_flags,
                                     std::uint64_t code_off,
                                     std::ifstream& input_file,
                                     std::uint64_t file_size,
                                     std::shared_ptr<DexTypes> dex_types)
        {
            this->method_id = method_id;
            this->access_flags = static_cast<DVMTypes::ACCESS_FLAGS>(access_flags);
            this->code_off = code_off;

            if (!parse_code_item(input_file, file_size, dex_types))
                throw exceptions::ParserReadingException("Error reading EncodedMethod");
        }

        EncodedMethod::~EncodedMethod() {}

        MethodID *EncodedMethod::get_method()
        {
            return method_id;
        }

        DVMTypes::ACCESS_FLAGS EncodedMethod::get_access_flags()
        {
            return access_flags;
        }

        std::uint64_t EncodedMethod::get_code_offset()
        {
            return code_off;
        }
    
        std::shared_ptr<CodeItemStruct> EncodedMethod::get_code_item()
        {
            return code_item;
        }

        bool EncodedMethod::parse_code_item(std::ifstream& input_file, std::uint64_t file_size, std::shared_ptr<DexTypes> dex_types)
        {
            auto current_offset = input_file.tellg();
            CodeItemStruct::code_item_struct_t code_item_struct;

            input_file.seekg(code_off);

            if (!KUNAI::read_data_file<CodeItemStruct::code_item_struct_t>(code_item_struct, sizeof(CodeItemStruct::code_item_struct_t), input_file))
                return false;

            if (code_item_struct.debug_info_off >= file_size)
                throw exceptions::OutOfBoundException("Error reading code_item_struct_t.debug_info_off out of file bound");

            code_item = std::make_shared<CodeItemStruct>(input_file, file_size, code_item_struct, dex_types);
            
            input_file.seekg(current_offset);
            return true;
        }
    }
}