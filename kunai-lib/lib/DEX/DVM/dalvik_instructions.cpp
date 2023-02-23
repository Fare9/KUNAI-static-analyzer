//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dalvik_instructions.cpp

#include "Kunai/DEX/DVM/dalvik_instructions.hpp"
#include "Kunai/Exceptions/disassembler_exception.hpp"

using namespace KUNAI::DEX;

Instruction10x::Instruction10x(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION10X, 2)
{
    if (op_codes[1] != 0)
        throw exceptions::DisassemblerException("Instruction10x high byte should be 0");
    
    op = op_codes[0];
}

Instruction12x::Instruction12x(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION12X, 2)
{
    op = op_codes[0];
    vA = (op_codes[1] & 0x0F);
    vB = (op_codes[1] & 0xF0)>>4;
}

Instruction11n::Instruction11n(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION11N, 2)
{
    op = op_codes[0];
    vA = op_codes[1] & 0x0F;
    nB = static_cast<std::int8_t>((op_codes[1] & 0xF0) >> 4);
}

Instruction11x::Instruction11x(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION11X, 2)
{
    op = op_codes[0];
    vAA = op_codes[1];
}

Instruction10t::Instruction10t(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION10T, 2)
{
    op = op_codes[0];
    nAA = static_cast<std::int8_t>(op_codes[1]);
}

Instruction20t::Instruction20t(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION20T, 4)
{
    if (op_codes[1] != 0)
        throw exceptions::DisassemblerException("Error reading Instruction20t padding must be 0");
    op = op_codes[0];
    nAAAA = *(reinterpret_cast<std::uint16_t*>(&op_codes[2]));
}

Instruction20bc::Instruction20bc(std::vector<uint8_t>& bytecode, std::size_t index)
    : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION20BC, 4)
{
    op = op_codes[0];
    nAA = op_codes[1];
    nBBBB = *(reinterpret_cast<std::uint16_t*>(&op_codes[2]));
}