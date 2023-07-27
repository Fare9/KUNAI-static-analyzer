#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction21s *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_16:
    {
        auto value = static_cast<std::int32_t>(instr->get_source());

        auto gen_value = builder.create<::mlir::arith::ConstantIntOp>(
            location,
            value,
            32);

        writeLocalVariable(analysis_context.current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_WIDE_16:
        {
            auto value = static_cast<std::int32_t>(instr->get_source());

            auto gen_value = builder.create<::mlir::arith::ConstantIntOp>(
                location,
                value,
                64);

            writeLocalVariable(analysis_context.current_basic_block, dest, gen_value);
        }
        break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction21s not supported");
        break;
    }
}
