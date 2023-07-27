//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file InstructionLifter.cpp
#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

namespace
{

    typedef void (*generator_func)(KUNAI::MjolnIR::Lifter *, KUNAI::DEX::Instruction *);

    template <class T>
    void gen_instruction(KUNAI::MjolnIR::Lifter *lifter, KUNAI::DEX::Instruction *instr)
    {
        lifter->gen_instruction(reinterpret_cast<T>(instr));
    }

    template<>
    void gen_instruction<KUNAI::DEX::PackedSwitch>(KUNAI::MjolnIR::Lifter *, KUNAI::DEX::Instruction *)
    {
        return;
    }

    template<>
    void gen_instruction<KUNAI::DEX::SparseSwitch>(KUNAI::MjolnIR::Lifter *, KUNAI::DEX::Instruction *)
    {
        return;
    }

    std::unordered_map<KUNAI::DEX::dexinsttype_t, generator_func> generators = {
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION23X, &gen_instruction<KUNAI::DEX::Instruction23x *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION12X, &gen_instruction<KUNAI::DEX::Instruction12x *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11X, &gen_instruction<KUNAI::DEX::Instruction11x *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22C, &gen_instruction<KUNAI::DEX::Instruction22c *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22T, &gen_instruction<KUNAI::DEX::Instruction22t *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21T, &gen_instruction<KUNAI::DEX::Instruction21t *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION10T, &gen_instruction<KUNAI::DEX::Instruction10t *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION20T, &gen_instruction<KUNAI::DEX::Instruction20t *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION30T, &gen_instruction<KUNAI::DEX::Instruction30t *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION10X, &gen_instruction<KUNAI::DEX::Instruction10x *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11N, &gen_instruction<KUNAI::DEX::Instruction11n *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21S, &gen_instruction<KUNAI::DEX::Instruction21s *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21H, &gen_instruction<KUNAI::DEX::Instruction21h *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION51L, &gen_instruction<KUNAI::DEX::Instruction51l *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION35C, &gen_instruction<KUNAI::DEX::Instruction35c *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21C, &gen_instruction<KUNAI::DEX::Instruction21c *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22X, &gen_instruction<KUNAI::DEX::Instruction22x *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION32X, &gen_instruction<KUNAI::DEX::Instruction32x *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31I, &gen_instruction<KUNAI::DEX::Instruction31i *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31C, &gen_instruction<KUNAI::DEX::Instruction31c *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22S, &gen_instruction<KUNAI::DEX::Instruction22s *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22B, &gen_instruction<KUNAI::DEX::Instruction22b *>},
        {KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31T, &gen_instruction<KUNAI::DEX::Instruction31t *>},
        {KUNAI::DEX::dexinsttype_t::DEX_PACKEDSWITCH, &gen_instruction<KUNAI::DEX::PackedSwitch *>},
        {KUNAI::DEX::dexinsttype_t::DEX_SPARSESWITCH, &gen_instruction<KUNAI::DEX::SparseSwitch *>},
    };
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction *instr)
{

    auto fIt = ::generators.find(instr->get_instruction_type());

    if (fIt != ::generators.end() && fIt->second != nullptr)
        fIt->second(this, instr);
    else
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: InstructionType not implemented");
}