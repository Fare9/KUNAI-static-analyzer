#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction10x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_RETURN_VOID:
        builder.create<::mlir::func::ReturnOp>(
            location);
        break;
    case KUNAI::DEX::TYPES::OP_NOP:
        builder.create<::mlir::KUNAI::MjolnIR::Nop>(
            location);
        break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction10x not supported");
        break;
    }
}