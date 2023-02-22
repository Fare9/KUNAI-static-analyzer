//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dalvik_opcodes.cpp
#include "Kunai/DEX/DVM/dalvik_opcodes.hpp"

using namespace KUNAI::DEX;

/// @brief Map of the opcodes and the name of the instructions
/// we will obtain its value from a definition file
static const std::unordered_map<TYPES::opcodes, std::string>
    opcodes_instruction_name =
        {
#define INST_NAME(OP, NAME) {OP, NAME},
#include "Kunai/DEX/DVM/dvm_inst_names.def"
};

static const std::unordered_map<TYPES::opcodes, TYPES::Kind>
    opcodes_instruction_kind =
        {
#define INST_KIND(OP, VAL) {OP, VAL},
#include "Kunai/DEX/DVM/dvm_inst_kind.def"
};

static const std::unordered_map<TYPES::opcodes, TYPES::Operation>
    opcodes_instruction_operation =
        {
#define INST_OP(OP, VAL) {OP, VAL},
#include "Kunai/DEX/DVM/dvm_inst_operation.def"
};

namespace
{
    std::string access_flags_to_str(TYPES::access_flags ac)
    {
        std::string access_flags = "";

        if (ac & TYPES::access_flags::ACC_PUBLIC)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_PUBLIC) + " ";

        if (ac & TYPES::access_flags::ACC_PRIVATE)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_PRIVATE) + " ";

        if (ac & TYPES::access_flags::ACC_PROTECTED)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_PROTECTED) + " ";

        if (ac & TYPES::access_flags::ACC_STATIC)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_FINAL) + " ";

        if (ac & TYPES::access_flags::ACC_SYNCHRONIZED)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_SYNCHRONIZED) + " ";

        if (ac & TYPES::access_flags::ACC_VOLATILE)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_VOLATILE) + " ";

        if (ac & TYPES::access_flags::ACC_BRIDGE)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_BRIDGE) + " ";

        if (ac & TYPES::access_flags::ACC_TRANSIENT)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_TRANSIENT) + " ";

        if (ac & TYPES::access_flags::ACC_VARARGS)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_VARARGS) + " ";

        if (ac & TYPES::access_flags::ACC_NATIVE)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_NATIVE) + " ";

        if (ac & TYPES::access_flags::ACC_INTERFACE)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_INTERFACE) + " ";

        if (ac & TYPES::access_flags::ACC_ABSTRACT)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_ABSTRACT) + " ";

        if (ac & TYPES::access_flags::ACC_STRICT)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_STRICT) + " ";

        if (ac & TYPES::access_flags::ACC_SYNTHETIC)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_SYNTHETIC) + " ";

        if (ac & TYPES::access_flags::ACC_ANNOTATION)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_ANNOTATION) + " ";

        if (ac & TYPES::access_flags::ACC_ENUM)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_ENUM) + " ";

        if (ac & TYPES::access_flags::ACC_CONSTRUCTOR)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_CONSTRUCTOR) + " ";

        if (ac & TYPES::access_flags::ACC_DECLARED_SYNCHRONIZED)
            access_flags += TYPES::ACCESS_FLAGS_STR.at(TYPES::access_flags::ACC_DECLARED_SYNCHRONIZED) + " ";

        if (!access_flags.empty())
            return access_flags.substr(0, access_flags.size() - 1);
        return "";
    }
} // namespace

const std::string &DalvikOpcodes::get_instruction_name(std::uint32_t instruction)
{
    auto it = opcodes_instruction_name.find(static_cast<TYPES::opcodes>(instruction));

    if (it == opcodes_instruction_name.end())
        return opcodes_instruction_name.at(TYPES::opcodes::OP_NONE);

    return it->second;
}

TYPES::Kind DalvikOpcodes::get_instruction_type(std::uint32_t instruction)
{
    auto it = opcodes_instruction_kind.find(static_cast<TYPES::opcodes>(instruction));

    if (it == opcodes_instruction_kind.end())
        return TYPES::Kind::NONE_KIND;

    return it->second;
}

TYPES::Operation DalvikOpcodes::get_instruction_operation(std::uint32_t instruction)
{
    auto it = opcodes_instruction_operation.find(static_cast<TYPES::opcodes>(instruction));

    if (it == opcodes_instruction_operation.end())
        return TYPES::Operation::NONE_OPCODE;

    return it->second;
}

const std::string &DalvikOpcodes::get_instruction_type_string(std::uint32_t instruction)
{
    auto kind = DalvikOpcodes::get_instruction_type(instruction);

    return TYPES::KindString.at(kind);
}

std::string DalvikOpcodes::get_method_access_flags(EncodedMethod *method)
{
    auto method_access_flags = method->get_access_flags();

    return ::access_flags_to_str(method_access_flags);
}

std::string DalvikOpcodes::get_field_access_flags(EncodedField* field)
{
    auto field_access_flags = field->get_access_flags();

    return ::access_flags_to_str(field_access_flags);
}

