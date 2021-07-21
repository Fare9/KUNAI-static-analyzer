#include "dex_dalvik_opcodes.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * @brief DalvikOpcodes constructor.
         * @param dex_parser: std::shared_ptr<DexParser> object to use in the object.
         */
        DalvikOpcodes::DalvikOpcodes(std::shared_ptr<DexParser> dex_parser)
        {
            this->dex_parser = dex_parser;
        }

        /**
         * @brief DalvikOpcodes destructor.
         */
        DalvikOpcodes::~DalvikOpcodes() {}

        /**
         * @brief find the instruction opcode in the map to get instruction name.
         * @param instruction: std::uint32_t instruction opcode.
         * @return std::string
         */
        std::string DalvikOpcodes::get_instruction_name(std::uint32_t instruction)
        {
            if (opcodes_instruction_name.find(instruction) == opcodes_instruction_name.end())
                return "";
            return opcodes_instruction_name[instruction];
        }

        /**
         * @brief find the instruction Kind given an instruction opcode.
         * @param instruction: std::uint32_t instruction opcode.
         * @return DVMTypes::Kind
         */
        DVMTypes::Kind DalvikOpcodes::get_instruction_type(std::uint32_t instruction)
        {
            if (opcodes_instruction_type.find(instruction) == opcodes_instruction_type.end())
                return DVMTypes::Kind::NONE_KIND;
            return opcodes_instruction_type[instruction];
        }

        /**
         * @brief find the instruction Operation given an instruction opcode.
         * @param instruction: std::uint32_t instruction opcode.
         * @return DVMTypes::Operation
         */
        DVMTypes::Operation DalvikOpcodes::get_instruction_operation(std::uint32_t instruction)
        {
            if (opcode_instruction_operation.find(instruction) == opcode_instruction_operation.end())
                return DVMTypes::Operation::NONE_OPCODE;
            return opcode_instruction_operation[instruction];
        }

        /**
         * @brief get instruction type as string.
         * @param instruction: std::uint32_t instruction opcode.
         * @return std::string
         */
        std::string DalvikOpcodes::get_instruction_type_str(std::uint32_t instruction)
        {
            DVMTypes::Kind kind = get_instruction_type(instruction);

            return KindString[kind];
        }

        /**
         * @brief get string* by id.
         * @param id: std::uint32_t id of string.
         * @return std::string*
         */
        std::string *DalvikOpcodes::get_dalvik_string_by_id(std::uint32_t id)
        {
            return dex_parser->get_strings()->get_string_from_order(id);
        }

        /**
         * @brief get Type* by id.
         * @param id: std::uint32_t id of the Type.
         * @return Type*
         */
        Type *DalvikOpcodes::get_dalvik_Type_by_id(std::uint32_t id)
        {
            return dex_parser->get_types()->get_type_from_order(id);
        }

        /**
         * @brief get FieldID* by id.
         * @param id: std::uint32_t of the FieldID.
         * @return FieldID*
         */
        FieldID *DalvikOpcodes::get_dalvik_field_by_id(std::uint32_t id)
        {
            return dex_parser->get_fields()->get_field_id_by_order(id);
        }

        /**
         * @brief get MethodID* by id.
         * @param id: std::uint32_t of the MethodID.
         * @return MethodID*
         */
        MethodID *DalvikOpcodes::get_dalvik_method_by_id(std::uint32_t id)
        {
            return dex_parser->get_methods()->get_method_by_order(id);
        }

        /**
         * @brief get ProtoID* by id.
         * @param id: std::uint32_t of the ProtoID.
         * @return ProtoID*
         */
        ProtoID *DalvikOpcodes::get_dalvik_proto_by_id(std::uint32_t id)
        {
            return dex_parser->get_protos()->get_proto_by_order(id);
        }

        /**
         * @brief get string by id.
         * @param id: std::uint32_t id of the string.
         * @return std::string
         */
        std::string DalvikOpcodes::get_dalvik_string_by_id_str(std::uint32_t id)
        {
            return *dex_parser->get_strings()->get_string_from_order(id);
        }
        
        /**
         * @brief Get a dalvil EncodedField by a given FieldID*
         * @param field: FieldID* field to obtain its std::shared_ptr<EncodedField>
         * @return std::shared_ptr<EncodedField>
         */
        std::shared_ptr<EncodedField> DalvikOpcodes::get_dalvik_encoded_field_by_fieldid(FieldID* field)
        {
            auto classes_def = dex_parser->get_classes_def_item();

            for (auto c = classes_def.begin(); c != classes_def.end(); c++)
            {
                if ((*c)->get_class_idx()->get_name() == reinterpret_cast<Class*>(field->get_class_idx())->get_name())
                {
                    auto fields = (*c)->get_class_data()->get_fields();

                    for (auto f = fields.begin(); f != fields.end(); f++)
                    {
                        if ((*f)->get_field() == field)
                        {
                            return *f;
                        }
                    }
                }
            }

            return nullptr;
        }

        /**
         * @brief get raw string from Type by id.
         * @param id: std::uint32_t id of the Type.
         * @return std::string
         */
        std::string DalvikOpcodes::get_dalvik_type_by_id_str(std::uint32_t id)
        {
            return dex_parser->get_types()->get_type_from_order(id)->get_raw();
        }

        /**
         * @brief get FieldID as string by id.
         * @param id: std::uint32_t id of the FieldID.
         * @return std::string
         */
        std::string DalvikOpcodes::get_dalvik_static_field_by_id_str(std::uint32_t id)
        {
            std::string field;

            field = dex_parser->get_fields()->get_field_id_by_order(id)->get_type_idx()->get_raw();
            field += " ";
            field += dex_parser->get_fields()->get_field_id_by_order(id)->get_class_idx()->get_raw();
            field += ".";
            field += *dex_parser->get_fields()->get_field_id_by_order(id)->get_name_idx();

            return field;
        }

        /**
         * @brief get MethodID as string by id.
         * @param id: std::uint32_t id of the MethodID.
         * @return std::string
         */
        std::string DalvikOpcodes::get_dalvik_method_by_id_str(std::uint32_t id)
        {
            std::string method;

            method = dex_parser->get_methods()->get_method_by_order(id)->get_method_class()->get_raw();
            method += "->";
            method += *dex_parser->get_methods()->get_method_by_order(id)->get_method_name();

            method += "(";

            size_t n_params = dex_parser->get_methods()->get_method_by_order(id)->get_method_prototype()->get_number_of_parameters();

            for (size_t i = 0; i < n_params; i++)
            {
                if (i == n_params - 1)
                    method += dex_parser->get_methods()->get_method_by_order(id)->get_method_prototype()->get_parameter_type_by_order(i)->get_raw();
                else
                    method += dex_parser->get_methods()->get_method_by_order(id)->get_method_prototype()->get_parameter_type_by_order(i)->get_raw() + ", ";
            }

            method += ")";

            method += dex_parser->get_methods()->get_method_by_order(id)->get_method_prototype()->get_return_idx()->get_raw();

            return method;
        }

        /**
         * @brief get ProtoID as string by id.
         * @param id: std::uint32_t id of the ProtoID.
         * @return std::string
         */
        std::string DalvikOpcodes::get_dalvik_proto_by_id_str(std::uint32_t id)
        {
            std::string proto;

            size_t n_params = dex_parser->get_protos()->get_proto_by_order(id)->get_number_of_parameters();

            proto = "(";

            for (size_t i = 0; i < n_params; i++)
            {
                if (i == n_params - 1)
                    proto += dex_parser->get_protos()->get_proto_by_order(id)->get_parameter_type_by_order(i)->get_raw();
                else
                    proto += proto += dex_parser->get_protos()->get_proto_by_order(id)->get_parameter_type_by_order(i)->get_raw() + ",";
            }

            proto += ")";

            proto += dex_parser->get_protos()->get_proto_by_order(id)->get_return_idx()->get_raw();

            return proto;
        }

        /**
         * @brief Method to get a string from the access flags.
         * @param acc_flag: value from enum DVMTypes::ACCESS_FLAGS.
         * @return std::string
         */
        std::string DalvikOpcodes::get_access_flags_string(DVMTypes::ACCESS_FLAGS acc_flag)
        {
            std::string flag = "";

            if (acc_flag & DVMTypes::ACC_PUBLIC)
                flag += "PUBLIC|";
            if (acc_flag & DVMTypes::ACC_PRIVATE)
                flag += "PRIVATE|";
            if (acc_flag & DVMTypes::ACC_PROTECTED)
                flag += "PROTECTED|";
            if (acc_flag & DVMTypes::ACC_STATIC)
                flag += "STATIC|";
            if (acc_flag & DVMTypes::ACC_FINAL)
                flag += "FINAL|";
            if (acc_flag & DVMTypes::ACC_SYNCHRONIZED)
                flag += "SYNCHRONIZED|";
            if (acc_flag & DVMTypes::ACC_VOLATILE)
                flag += "VOLATILE|";
            if (acc_flag & DVMTypes::ACC_BRIDGE)
                flag += "BRIDGE|";
            if (acc_flag & DVMTypes::ACC_TRANSIENT)
                flag += "TRANSIENT|";
            if (acc_flag & DVMTypes::ACC_VARARGS)
                flag += "VARARGS|";
            if (acc_flag & DVMTypes::ACC_NATIVE)
                flag += "NATIVE|";
            if (acc_flag & DVMTypes::ACC_INTERFACE)
                flag += "INTERFACE|";
            if (acc_flag & DVMTypes::ACC_ABSTRACT)
                flag += "ABSTRACT|";
            if (acc_flag & DVMTypes::ACC_STRICT)
                flag += "STRICT|";
            if (acc_flag & DVMTypes::ACC_SYNTHETIC)
                flag += "SYNTHETIC|";
            if (acc_flag & DVMTypes::ACC_ANNOTATION)
                flag += "ANNOTATION|";
            if (acc_flag & DVMTypes::ACC_ENUM)
                flag += "ENUM|";
            if (acc_flag & DVMTypes::ACC_CONSTRUCTOR)
                flag += "CONSTRUCTOR|";
            if (acc_flag & DVMTypes::ACC_DECLARED_SYNCHRONIZED)
                flag += "DECLARED_SYNCHRONIZED|";

            flag.pop_back();

            return flag;
        }
    }
}