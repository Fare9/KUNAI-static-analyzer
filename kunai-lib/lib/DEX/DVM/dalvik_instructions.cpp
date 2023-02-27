//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dalvik_instructions.cpp

#include "Kunai/DEX/DVM/dalvik_instructions.hpp"
#include "Kunai/Exceptions/invalidinstruction_exception.hpp"

using namespace KUNAI::DEX;

Instruction10x::Instruction10x(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION10X, 2)
{
    if (op_codes[1] != 0)
        throw exceptions::InvalidInstructionException("Instruction10x high byte should be 0");

    op = op_codes[0];
}

Instruction12x::Instruction12x(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION12X, 2)
{
    op = op_codes[0];
    vA = (op_codes[1] & 0x0F);
    vB = (op_codes[1] & 0xF0) >> 4;
}

Instruction11n::Instruction11n(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION11N, 2)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    nB = static_cast<std::int8_t>((op_codes[1] & 0xF0) >> 4);
}

Instruction11x::Instruction11x(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION11X, 2)
{
    op = op_codes[0];
    vAA = op_codes[1];
}

Instruction10t::Instruction10t(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION10T, 2)
{
    op = op_codes[0];
    nAA = static_cast<std::int8_t>(op_codes[1]);
}

Instruction20t::Instruction20t(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION20T, 4)
{
    if (op_codes[1] != 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction20t padding must be 0");
    op = op_codes[0];
    nAAAA = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
}

Instruction20bc::Instruction20bc(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION20BC, 4)
{
    op = op_codes[0];
    nAA = op_codes[1];
    nBBBB = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
}

Instruction22x::Instruction22x(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION22X, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&op_codes[2]));
}

Instruction21t::Instruction21t(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION21T, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&op_codes[2]));

    if (nBBBB == 0)
        throw exceptions::InvalidInstructionException("Error reading Instruction21t offset cannot be 0");
}

Instruction21h::Instruction21h(std::vector<uint8_t> &bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION21H, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    std::int16_t nBBBB_aux = *(reinterpret_cast<std::int16_t *>(&op_codes[2]));

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
        source_str = parser->get_strings().get_string_by_id(iBBBB);
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

Instruction23x::Instruction23x(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION23X, 4)
{
    op = op_codes[0];
    vAA = op_codes[1];
    vBB = op_codes[2];
    vCC = op_codes[3];
}