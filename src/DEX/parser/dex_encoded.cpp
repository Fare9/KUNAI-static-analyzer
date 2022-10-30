#include "KUNAI/DEX/parser/dex_encoded.hpp"

namespace KUNAI
{
    namespace DEX
    {
        EncodedValue::EncodedValue(std::ifstream &input_file)
        {
            std::uint8_t value, value_type, value_args;

            if (!KUNAI::read_data_file<std::uint8_t>(value, sizeof(std::uint8_t), input_file))
                throw exceptions::ParserReadingException("Error reading EncodedValue 'value' from file");

            value_type = value & 0x1f;
            value_args = ((value & 0xe0) >> 5);

            switch (value_type)
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
                encodedvalue_t aux;
                for (size_t i = 0; i < size; i++)
                {
                    aux = std::make_unique<EncodedValue>(input_file);
                    array.push_back(std::move(aux));
                }
                break;
            }
            case DVMTypes::VALUE_ANNOTATION:
            {
                annotation = std::make_unique<EncodedAnnotation>(input_file);
                break;
            }
            case DVMTypes::VALUE_NULL:
            case DVMTypes::VALUE_BOOLEAN:
                break;
            }
        }

        /***
         * EncodedArray
         */
        EncodedArray::EncodedArray(std::ifstream &input_file)
        {
            size = KUNAI::read_uleb128(input_file);
            encodedvalue_t encoded_value;

            for (size_t i = 0; i < size; i++)
            {
                encoded_value = std::make_unique<EncodedValue>(input_file);
                values.push_back(std::move(encoded_value));
            }
        }

        /***
         * EncodedArrayItem
         */
        EncodedArrayItem::EncodedArrayItem(std::ifstream &input_file)
        {
            this->array = std::make_unique<EncodedArray>(input_file);
        }

        /***
         * EncodedField
         */
        EncodedField::EncodedField(FieldID *field_idx, std::uint64_t access_flags) : field_idx(field_idx),
                                                                                     access_flags(static_cast<DVMTypes::ACCESS_FLAGS>(access_flags))
        {
        }

        /***
         * EncodedTypePair
         */
        EncodedTypePair::EncodedTypePair(std::uint64_t type_idx,
                                         std::uint64_t addr,
                                         DexTypes *dex_types)
        {
            this->type_idx = std::make_pair(type_idx, dex_types->get_type_from_order(type_idx));
            this->addr = addr;
        }

        Type *EncodedTypePair::get_exception_type()
        {
            return type_idx.second;
        }

        /***
         * EncodedCatchHandler
         */
        EncodedCatchHandler::EncodedCatchHandler(std::ifstream &input_file,
                                                 std::uint64_t file_size,
                                                 DexTypes *dex_types)
        {
            if (!parse_encoded_type_pairs(input_file, file_size, dex_types))
                throw exceptions::ParserReadingException("Error reading EncodedTypePair from EncodedCatchHandler");
        }

        bool EncodedCatchHandler::has_explicit_typed_catches()
        {
            if (encoded_type_pair_size >= 0)
                return true; // user should check size of handler
            return false;
        }

        EncodedTypePair *EncodedCatchHandler::get_handler_by_pos(std::uint64_t pos) const
        {
            if (pos >= handlers.size())
                return nullptr;
            return handlers[pos].get();
        }

        bool EncodedCatchHandler::parse_encoded_type_pairs(std::ifstream &input_file,
                                                           std::uint64_t file_size,
                                                           DexTypes *dex_types)
        {
            auto current_offset = input_file.tellg();
            offset = current_offset;
            std::uint64_t type_idx, addr;
            encodedtypepair_t encoded_type_pair;

            encoded_type_pair_size = KUNAI::read_sleb128(input_file);

            for (size_t i = 0; i < std::abs(encoded_type_pair_size); i++)
            {
                type_idx = KUNAI::read_uleb128(input_file);

                addr = KUNAI::read_uleb128(input_file);

                encoded_type_pair = std::make_unique<EncodedTypePair>(type_idx, addr, dex_types);
                handlers.push_back(std::move(encoded_type_pair));
            }

            if (encoded_type_pair_size <= 0)
            {
                catch_all_addr = KUNAI::read_uleb128(input_file);
            }

            return true;
        }

        /***
         * TryItem
         */
        TryItem::TryItem(try_item_struct_t try_item_struct) : try_item_struct(try_item_struct)
        {
        }

        /***
         * CodeItemStruct
         */
        CodeItemStruct::CodeItemStruct(std::ifstream &input_file,
                                       std::uint64_t file_size,
                                       code_item_struct_t code_item,
                                       DexTypes *dex_types) : code_item(code_item)
        {
            if (!parse_code_item_struct(input_file, file_size, dex_types))
                throw exceptions::ParserReadingException("Error reading CodeItemStruct");
        }

        TryItem* CodeItemStruct::get_try_item_by_pos(std::uint64_t pos)
        {
            if (pos >= try_items.size())
                return nullptr;
            return try_items[pos].get();
        }

        std::uint16_t CodeItemStruct::get_raw_instruction_by_pos(std::uint16_t pos)
        {
            if (pos >= instructions_raw.size())
                return 0;
            return instructions_raw[pos] | (instructions_raw[pos + 1] << 8);
        }

        EncodedCatchHandler * CodeItemStruct::get_encoded_catch_handler_by_pos(std::uint64_t pos)
        {
            if (pos >= encoded_catch_handler_list.size())
                return nullptr;
            return encoded_catch_handler_list[pos].get();
        }

        bool CodeItemStruct::parse_code_item_struct(std::ifstream &input_file,
                                                    std::uint64_t file_size,
                                                    DexTypes *dex_types)
        {
            auto current_offset = input_file.tellg();

            size_t i;
            std::uint16_t instruction;
            TryItem::try_item_struct_t try_item_struct;
            tryitem_t try_item;
            std::uint64_t encoded_catch_handler_list_size;
            encodedcatchhandler_t encoded_catch_handler;

            for (i = 0; i < code_item.insns_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint16_t>(instruction, sizeof(std::uint16_t), input_file))
                    return false;
                instructions_raw.push_back(instruction & 0xff);
                instructions_raw.push_back((instruction >> 8) & 0xff);
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

                    try_item = std::make_unique<TryItem>(try_item_struct);
                    try_items.push_back(std::move(try_item));
                }

                // necessary for exception calculation
                encoded_catch_handler_list_offset = input_file.tellg();

                encoded_catch_handler_list_size = KUNAI::read_uleb128(input_file);

                for (i = 0; i < encoded_catch_handler_list_size; i++)
                {
                    encoded_catch_handler = std::make_unique<EncodedCatchHandler>(input_file, file_size, dex_types);
                    encoded_catch_handler_list.push_back(std::move(encoded_catch_handler));
                }
            }

            input_file.seekg(current_offset);
            return true;
        }

        /***
         * EncodedMethod
         */
        EncodedMethod::EncodedMethod(MethodID* method_id,
                                     std::uint64_t access_flags,
                                     std::uint64_t code_off,
                                     std::ifstream &input_file,
                                     std::uint64_t file_size,
                                     DexTypes* dex_types) : method_id(method_id),
                                                              access_flags(static_cast<DVMTypes::ACCESS_FLAGS>(access_flags)),
                                                              code_off(code_off)
        {
            if (!parse_code_item(input_file, file_size, dex_types))
                throw exceptions::ParserReadingException("Error reading EncodedMethod");
        }

        std::string EncodedMethod::full_name()
        {
            std::string class_name = "";

            switch (method_id->get_method_class()->get_type())
            {
            case Type::FUNDAMENTAL:
                class_name = reinterpret_cast<Fundamental*>(method_id->get_method_class())->print_fundamental_type();
                break;
            case Type::CLASS:
                class_name = reinterpret_cast<Class*>(method_id->get_method_class())->get_name();
                break;
            default:
                class_name = method_id->get_method_class()->get_raw();
                break;
            }

            return class_name + "->" +
                   *method_id->get_method_name() + " " +
                   method_id->get_method_prototype()->get_proto_str();
        }

        bool EncodedMethod::parse_code_item(std::ifstream &input_file, std::uint64_t file_size, DexTypes* dex_types)
        {
            auto current_offset = input_file.tellg();
            CodeItemStruct::code_item_struct_t code_item_struct;

            if (code_off > 0)
            {
                input_file.seekg(code_off);

                if (!KUNAI::read_data_file<CodeItemStruct::code_item_struct_t>(code_item_struct, sizeof(CodeItemStruct::code_item_struct_t), input_file))
                    return false;

                if (code_item_struct.debug_info_off >= file_size)
                    throw exceptions::OutOfBoundException("Error reading code_item_struct_t.debug_info_off out of file bound");

                code_item = std::make_unique<CodeItemStruct>(input_file, file_size, code_item_struct, dex_types);
            }

            input_file.seekg(current_offset);
            return true;
        }

        /**
         * AnnotationElement
         */
        AnnotationElement::AnnotationElement(std::ifstream &input_file)
        {
            name_idx = read_uleb128(input_file);
            value = std::make_unique<EncodedValue>(input_file);
        }

        /**
         * EncodedAnnotation
         */
        EncodedAnnotation::EncodedAnnotation(std::ifstream &input_file)
        {
            type_idx = read_uleb128(input_file);
            size = read_uleb128(input_file);

            annotationelement_t annotation_element;

            for (size_t i = 0; i < size; i++)
            {
                annotation_element = std::make_unique<AnnotationElement>(input_file);
                elements.push_back(std::move(annotation_element));
            }
        }
    }
}