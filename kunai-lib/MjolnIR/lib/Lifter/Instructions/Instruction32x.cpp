#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction32x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_MOVE_16:
    case KUNAI::DEX::TYPES::OP_MOVE_WIDE_16:
    case KUNAI::DEX::TYPES::OP_MOVE_OBJECT_16:
    {
        auto src_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::MoveOp>(
            location,
            src_value.getType(),
            src_value);

        writeLocalVariable(analysis_context.current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction32x not supported");
        break;
    }
}