//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file InstructionLifter.cpp
#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction *instr)
{
    switch (instr->get_instruction_type())
    {
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION23X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction23x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION12X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction12x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction11x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION10T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction10t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION20T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction20t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION30T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction30t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION10X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction10x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11N:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction11n *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21S:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21s *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21H:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21h *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION51L:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction51l *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION35C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction35c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION32X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction32x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31I:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction31i *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction31c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22S:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22s *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22B:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22b *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction31t *>(instr));
        break;
    /// no need to implement these
    case KUNAI::DEX::dexinsttype_t::DEX_PACKEDSWITCH:
    case KUNAI::DEX::dexinsttype_t::DEX_SPARSESWITCH:
        break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: InstructionType not implemented");
    }
}