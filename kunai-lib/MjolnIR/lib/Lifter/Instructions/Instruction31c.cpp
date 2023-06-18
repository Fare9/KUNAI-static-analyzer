#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction31c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_CONST_STRING_JUMBO:
    {
        auto str_value = instr->get_string_value();
        auto str_ref = instr->get_string_idx();

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadString>(
            location,
            strObjectType,
            str_value,
            str_ref);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction31c not supported");
        break;
    }
}
