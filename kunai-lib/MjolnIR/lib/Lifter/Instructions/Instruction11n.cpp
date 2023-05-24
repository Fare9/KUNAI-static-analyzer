#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction11n *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_CONST_4:
    {
        auto value = instr->get_source();

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadValue>(
            location,
            byteType,
            value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction11n not supported");
        break;
    }
}
