#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction10t *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_GOTO:
    {
        auto offset = instr->get_jump_offset();
        auto target_idx = instr->get_address() + (offset * 2);

        auto target_block = analysis_context.current_method->get_basic_blocks().get_basic_block_by_idx(target_idx);

        builder.create<::mlir::cf::BranchOp>(
            location,
            map_blocks[target_block]);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction10t not supported or not recognized.");
        break;
    }
}