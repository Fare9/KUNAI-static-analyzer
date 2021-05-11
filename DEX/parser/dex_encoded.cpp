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
         * EncodedMethod
         */
        EncodedMethod::EncodedMethod(MethodID *method_id,
                                     std::uint64_t access_flags,
                                     std::uint64_t code_off)
        {
            this->method_id = method_id;
            this->access_flags = static_cast<DVMTypes::ACCESS_FLAGS>(access_flags);
            this->code_off = code_off;
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
    }
}