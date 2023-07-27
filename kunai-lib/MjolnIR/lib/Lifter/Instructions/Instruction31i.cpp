#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction31i *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_CONST:
    case KUNAI::DEX::TYPES::OP_CONST_WIDE_32:
    {
        /// for the moment set destination type as a long,
        /// we need to think a better algorithm

        auto value = instr->get_source_float();

        auto gen_value = builder.create<::mlir::arith::ConstantFloatOp>(
            location,
            ::mlir::APFloat(value),
            ::mlir::Float32Type::get(&context)
        );

        writeLocalVariable(analysis_context.current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction31i not supported");
        break;
    }
}
