//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dalvik_instructions.cpp

#include "Kunai/DEX/DVM/dalvik_instructions.hpp"
#include "Kunai/Exceptions/invalidinstruction_exception.hpp"

using namespace KUNAI::DEX;

Instruction10x::Instruction10x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION10X, 2)
{
    if (op_codes[1] != 0)
        throw exceptions::InvalidInstructionException("Instruction10x high byte should be 0", 2);

    op = op_codes[0];
}

Instruction12x::Instruction12x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION12X, 2)
{
    op = op_codes[0];
    vA = (op_codes[1] & 0x0F);
    vB = (op_codes[1] & 0xF0) >> 4;
}

Instruction11n::Instruction11n(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION11N, 2)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    nB = static_cast<std::int8_t>((op_codes[1] & 0xF0) >> 4);
}

Instruction11x::Instruction11x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION11X, 2)
{
    op = op_codes[0];
    vAA = op_codes[1];
}

Instruction10t::Instruction10t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION10T, 2)
{
    op = op_codes[0];
    nAA = static_cast<std::int8_t>(op_codes[1]);
}

Instruction20t::Instruction20t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION20T, 4)
{
    if (op_codes[1] != 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction20t padding must be 0", 4);
    op = op_codes[0];
    nAAAA = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
}

Instruction20bc::Instruction20bc(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION20BC, 4)
{
    op = op_codes[0];
    nAA = op_codes[1];
    nBBBB = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
}

Instruction22x::Instruction22x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22X, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
}

Instruction21t::Instruction21t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION21T, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&op_codes[2]));

    if (nBBBB == 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction21t offset cannot be 0", 4);
}

Instruction21s::Instruction21s(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION21S, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t*>(&op_codes[2]));
}

Instruction21h::Instruction21h(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION21H, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    std::int16_t const nBBBB_aux = *(reinterpret_cast<std::int16_t *>(&op_codes[2]));

    switch (op)
    {
    case TYPES::opcodes::OP_CONST_HIGH16:
        // const/high16 vAA, #+BBBB0000
        nBBBB = static_cast<std::int64_t>(nBBBB_aux) << 16;
        break;
    case TYPES::opcodes::OP_CONST_WIDE_HIGH16:
        // const-wide/high16 vAA, #+BBBB000000000000
        nBBBB = static_cast<std::int64_t>(nBBBB_aux) << 48;
        break;
    default:
        nBBBB = static_cast<std::int16_t>(nBBBB_aux);
    }
}

Instruction21c::Instruction21c(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION21C, 4), parser(parser)
{
    op = op_codes[0];
    vAA = op_codes[1];
    iBBBB = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));

    /// The instruction has a kind of operation depending
    /// on the op code, check it, and use it wisely
    switch (get_kind())
    {
    case TYPES::Kind::STRING:
        is_str = true;
        source_str = "\"" + parser->get_strings().get_string_by_id(iBBBB) + "\"";
        break;
    case TYPES::Kind::TYPE:
    {
        is_type = true;
        auto type = parser->get_types().get_type_from_order(iBBBB);
        source_str = type->pretty_print();
        if (type->get_type() == DVMType::FUNDAMENTAL)
            is_fundamental = true;
        else if (type->get_type() == DVMType::CLASS)
            is_class = true;
        else if (type->get_type() == DVMType::ARRAY)
            is_array = true;
        else
            is_unknown = true;
    }
    break;
    case TYPES::Kind::FIELD:
        is_field = true;
        source_str = parser->get_fields().get_field(iBBBB)->pretty_field();
        break;
    case TYPES::Kind::METH:
        is_method = true;
        source_str = parser->get_methods().get_method(iBBBB)->pretty_method();
        break;
    case TYPES::Kind::PROTO:
        is_proto = true;
        source_str = parser->get_protos().get_proto_by_order(iBBBB)->get_shorty_idx();
        break;
    }
}

Instruction23x::Instruction23x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION23X, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    vBB = op_codes[2];
    vCC = op_codes[3];
}

Instruction22b::Instruction22b(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22B, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    vBB = op_codes[2];
    nCC = static_cast<std::int8_t>(op_codes[3]);
}

Instruction22t::Instruction22t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22T, 4)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    vB = (op_codes[1] & 0xF0) >> 4;
    nCCCC = *(reinterpret_cast<std::int16_t *>(&op_codes[2]));

    if (nCCCC == 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction22t offset cannot be 0", 4);
}

Instruction22s::Instruction22s(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22S, 4)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    vB = (op_codes[1] & 0xF0) >> 4;
    nCCCC = *(reinterpret_cast<std::int16_t *>(&op_codes[2]));
}

Instruction22c::Instruction22c(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22C, 4), parser(parser)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    vB = (op_codes[1] & 0xF0) >> 4;
    iCCCC = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));

    /// as in Instruction21c we have a type for the instruction
    switch (get_kind())
    {
    case TYPES::Kind::FIELD:
        is_field = true;
        iCCCC_str = parser->get_fields().get_field(iCCCC)->pretty_field();
        break;
    case TYPES::Kind::TYPE:
        is_type = true;
        iCCCC_str = parser->get_types().get_type_from_order(iCCCC)->pretty_print();
        break;
    default:
        iCCCC_str = std::to_string(iCCCC);
        break;
    }
}

Instruction22cs::Instruction22cs(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22CS, 4), parser(parser)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    vB = (op_codes[1] & 0xF0) >> 4;
    iCCCC = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));

    switch (get_kind())
    {
    case TYPES::Kind::FIELD:
        is_field = true;
        iCCCC_str = parser->get_fields().get_field(iCCCC)->pretty_field();
        break;
    default:
        iCCCC_str = std::to_string(iCCCC);
        break;
    }
}

Instruction30t::Instruction30t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION30T, 6)
{
    if (op_codes[1] != 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction30t padding must be 0", 6);

    op = op_codes[0];
    nAAAAAAAA = *(reinterpret_cast<std::int32_t *>(&op_codes[2]));

    if (nAAAAAAAA == 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction30t offset cannot be 0", 6);
}

Instruction32x::Instruction32x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION32X, 6)
{
    if (op_codes[1] != 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction32x padding must be 0", 6);

    op = op_codes[0];
    vAAAA = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&op_codes[4]));
}

Instruction31i::Instruction31i(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION31I, 6)
{
    op = op_codes[0];
    vAA = op_codes[1];
    nBBBBBBBB = *(reinterpret_cast<std::uint32_t *>(&op_codes[2]));
}

Instruction31t::Instruction31t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION31T, 6)
{
    op = op_codes[0];
    vAA = op_codes[1];
    nBBBBBBBB = *(reinterpret_cast<std::int32_t *>(&op_codes[2]));

    switch (op)
    {
    case TYPES::opcodes::OP_PACKED_SWITCH:
        type_of_switch = PACKED_SWITCH;
        break;
    case TYPES::opcodes::OP_SPARSE_SWITCH:
        type_of_switch = SPARSE_SWITCH;
        break;
    default:
        type_of_switch = NONE_SWITCH;
        break;
    }
}

Instruction31c::Instruction31c(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION31C, 6)
{
    op = op_codes[0];
    vAA = op_codes[1];
    iBBBBBBBB = *(reinterpret_cast<std::uint32_t *>(&op_codes[2]));
    str_value = parser->get_strings().get_string_by_id(iBBBBBBBB);
}

Instruction35c::Instruction35c(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION35C, 6), parser(parser)
{
    /// for reading the registers
    std::uint8_t reg[5];

    op = op_codes[0];
    array_size = (op_codes[1] & 0xF0) >> 4;
    type_index = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));

    /// assign the values to the registers
    reg[4] = op_codes[1] & 0x0F;
    reg[0] = op_codes[4] & 0x0F;
    reg[1] = (op_codes[4] & 0xF0) >> 4;
    reg[2] = op_codes[5] & 0x0F;
    reg[3] = (op_codes[5] & 0xF0) >> 4;

    if (array_size > 5)
        throw exceptions::InvalidInstructionException("Error in array size of Instruction35c, cannot be greater than 5", 6);

    for (size_t I = 0; I < array_size; ++I)
        registers.push_back(reg[I]);

    switch (get_kind())
    {
    case TYPES::Kind::TYPE:
        is_type = true;
        type_str = parser->get_types().get_type_from_order(type_index)->pretty_print();
        break;
    case TYPES::Kind::METH:
        is_method = true;
        type_str = parser->get_methods().get_method(type_index)->pretty_method();
        break;
    /// others I don't know how to manage...
    default:
        type_str = std::to_string(type_index);
    }
}

Instruction3rc::Instruction3rc(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION3RC, 6), parser(parser)
{
    std::uint16_t vCCCC;
    op = op_codes[0];
    array_size = op_codes[1];
    index = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
    vCCCC = *(reinterpret_cast<std::uint16_t *>(&op_codes[4]));

    /// assign the registers starting by vCCCC
    for (std::uint16_t I = vCCCC, E = vCCCC + array_size;
         I < E; ++I)
    {
        registers.push_back(I);
    }

    switch (get_kind())
    {
    case TYPES::Kind::TYPE:
        is_type = true;
        index_str = parser->get_types().get_type_from_order(index)->pretty_print();
        break;
    case TYPES::Kind::METH:
        is_method = true;
        index_str = parser->get_methods().get_method(index)->pretty_method();
        break;
    /// other maybe needs to be managed
    default:
        index_str = std::to_string(index);
    }
}

Instruction45cc::Instruction45cc(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION45CC, 8)
{
    std::uint8_t regC, regD, regE, regF, regG;

    op = op_codes[0];
    reg_count = (op_codes[1] & 0xF0) >> 4;
    regG = op_codes[1] & 0x0F;
    method_reference = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
    regD = (op_codes[4] & 0xF0) >> 4;
    regC = op_codes[4] & 0x0F;
    regF = (op_codes[5] & 0xF0) >> 4;
    regE = op_codes[5] & 0x0F;
    prototype_reference = *(reinterpret_cast<std::uint16_t *>(&op_codes[8]));

    if (reg_count > 5)
        throw exceptions::InvalidInstructionException("Error in reg_count from Instruction45cc cannot be greater than 5", 8);

    if (method_reference >= parser->get_methods().get_number_of_methods())
        throw exceptions::InvalidInstructionException("Error method reference out of bound in Instruction45cc", 8);

    if (prototype_reference >= parser->get_protos().get_number_of_protos())
        throw exceptions::InvalidInstructionException("Error prototype reference out of bound in Instruction45cc", 8);

    if (reg_count > 0)
        registers.push_back(regC);
    if (reg_count > 1)
        registers.push_back(regD);
    if (reg_count > 2)
        registers.push_back(regE);
    if (reg_count > 3)
        registers.push_back(regF);
    if (reg_count > 4)
        registers.push_back(regG);
    
    method_id = parser->get_methods().get_method(method_reference);
    proto_id = parser->get_protos().get_proto_by_order(prototype_reference);
}

Instruction4rcc::Instruction4rcc(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION4RCC, 8)
{
    std::uint16_t vCCCC;

    op = op_codes[0];
    reg_count = op_codes[1];
    method_reference = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
    vCCCC = *(reinterpret_cast<std::uint16_t *>(&op_codes[4]));
    prototype_reference = *(reinterpret_cast<std::uint16_t *>(&op_codes[6]));

    if (method_reference >= parser->get_methods().get_number_of_methods())
        throw exceptions::InvalidInstructionException("Error method reference out of bound in Instruction4rcc", 8);

    if (prototype_reference >= parser->get_protos().get_number_of_protos())
        throw exceptions::InvalidInstructionException("Error prototype reference out of bound in Instruction4rcc", 8);
    
    for (std::uint16_t I = vCCCC, E = vCCCC+reg_count; I < E; ++I)
        registers.push_back(I);

    method_id = parser->get_methods().get_method(method_reference);
    prototype_id = parser->get_protos().get_proto_by_order(prototype_reference);
}

Instruction51l::Instruction51l(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION51L, 10)
{
    op = op_codes[0];
    vAA = op_codes[1];
    nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::int64_t*>(&op_codes[2]));
}

PackedSwitch::PackedSwitch(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_PACKEDSWITCH, 8)
{
    std::int32_t aux;

    op = *(reinterpret_cast<std::uint16_t*>(&op_codes[0]));
    size = *(reinterpret_cast<std::uint16_t*>(&op_codes[2]));
    first_key = *(reinterpret_cast<std::int32_t*>(&op_codes[4]));

    // because the instruction is larger, we have to 
    // re-accomodate the op_codes span and the length
    // we have to increment it 
    length += (size*4);

    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    // now read the targets
    auto multiplier = sizeof(std::int32_t);
    for (size_t I = 0; I < size; ++I)
    {
        aux = *(reinterpret_cast<std::int32_t*>(&op_codes[8+(I*multiplier)]));
        targets.push_back(aux);
    }
}

SparseSwitch::SparseSwitch(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_SPARSESWITCH, 4)
{
    std::int32_t aux_key, aux_target;

    op = *(reinterpret_cast<std::uint16_t*>(&op_codes[0]));
    size = *(reinterpret_cast<std::uint16_t*>(&op_codes[2]));

    // now we have to do as before, we have to set the appropiate
    // length and also fix the span object
    // the length is the number of keys and targets multiplied by
    // the size of each one
    length += (sizeof(std::int32_t) * size)*2;
    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    auto base_targets = 4 + sizeof(std::int32_t) * size;
    auto multiplier = sizeof(std::int32_t);

    for (size_t I = 0; I < size; ++I)
    {
        aux_key = *(reinterpret_cast<std::int32_t*>(&op_codes[4 + I*multiplier]));
        aux_target = *(reinterpret_cast<std::int32_t*>(&op_codes[base_targets + I*multiplier]));

        keys_targets.push_back({aux_key, aux_target});
    }
}

FillArrayData::FillArrayData(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser)
    : Instruction(bytecode, index, dexinsttype_t::DEX_FILLARRAYDATA, 8)
{
    std::uint8_t aux;

    op = *(reinterpret_cast<std::uint16_t*>(&op_codes[0]));
    element_width =  *(reinterpret_cast<std::uint16_t*>(&op_codes[2]));
    size = *(reinterpret_cast<std::uint32_t*>(&op_codes[4]));

    // again we have to fix the length of the instruction
    // and also the opcodes
    auto buff_size = (size*element_width);
    length += buff_size;
    if (buff_size % 2 != 0)
        length += 1;
    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};
    
    for (size_t I = 0; I < buff_size; ++I)
    {
        data.push_back(op_codes[8+I]);
    }
}