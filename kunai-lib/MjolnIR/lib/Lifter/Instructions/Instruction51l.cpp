#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction51l *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    /// we will take the registers as big enough
    /// for storing a 64-bit value
    auto dest_reg = instr->get_first_register();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_CONST_WIDE:
    {
        auto value = instr->get_wide_value_as_double();

        auto gen_value = builder.create<::mlir::arith::ConstantFloatOp>(
            location,
            ::mlir::APFloat(value),
            mlir::Float64Type::get(&context));

        writeLocalVariable(current_basic_block, dest_reg, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction51l not supported");
        break;
    }
}
