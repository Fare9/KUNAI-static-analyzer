#include "dex_dalvik_opcodes.hpp"

namespace KUNAI
{
    namespace DEX
    {

        DalvikOpcodes::DalvikOpcodes(std::shared_ptr<DexParser> dex_parser)
        {
            this->dex_parser = dex_parser;
        }

        DalvikOpcodes::~DalvikOpcodes() {}

        std::string DalvikOpcodes::get_instruction_name(std::uint32_t instruction)
        {
            if (opcodes_instruction_name.find(instruction) == opcodes_instruction_name.end())
                return "";
            return opcodes_instruction_name[instruction];
        }

        DVMTypes::Kind DalvikOpcodes::get_instruction_type(std::uint32_t instruction)
        {
            if (opcodes_instruction_type.find(instruction) == opcodes_instruction_type.end())
                return DVMTypes::Kind::NONE;
            return opcodes_instruction_type[instruction];
        }

        std::string DalvikOpcodes::get_instruction_type_str(std::uint32_t instruction)
        {
            DVMTypes::Kind kind = get_instruction_type(instruction);

            return KindString[kind];
        }

        std::string *DalvikOpcodes::get_dalvik_string_by_id(std::uint32_t id)
        {
            return dex_parser->get_strings()->get_string_from_order(id);
        }

        Type *DalvikOpcodes::get_dalvik_Type_by_id(std::uint32_t id)
        {
            return dex_parser->get_types()->get_type_from_order(id);
        }

        FieldID *DalvikOpcodes::get_dalvik_field_by_id(std::uint32_t id)
        {
            return dex_parser->get_fields()->get_field_id_by_order(id);
        }

        MethodID *DalvikOpcodes::get_dalvik_method_by_id(std::uint32_t id)
        {
            return dex_parser->get_methods()->get_method_by_order(id);
        }

        ProtoID *DalvikOpcodes::get_dalvik_proto_by_id(std::uint32_t id)
        {
            return dex_parser->get_protos()->get_proto_by_order(id);
        }

        std::string DalvikOpcodes::get_dalvik_string_by_id_str(std::uint32_t id)
        {
            return *dex_parser->get_strings()->get_string_from_order(id);
        }

        std::string DalvikOpcodes::get_dalvik_type_by_id_str(std::uint32_t id)
        {
            return dex_parser->get_types()->get_type_from_order(id)->get_raw();
        }

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
    }
}