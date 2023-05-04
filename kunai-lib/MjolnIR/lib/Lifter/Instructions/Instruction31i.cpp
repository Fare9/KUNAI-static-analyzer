#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction31i *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    mlir::Type dest_type;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_CONST:
    case KUNAI::DEX::TYPES::OP_CONST_WIDE_32:
    {
        /// for the moment set destination type as a long,
        /// we need to think a better algorithm
        if (!dest_type)
            dest_type = longType;

        auto value = static_cast<std::int64_t>(instr->get_source());

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadValue>(
            location,
            dest_type,
            value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction31i not supported");
        break;
    }
}
