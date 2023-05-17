#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction21h *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    mlir::Type dest_type;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_HIGH16:
        if (!dest_type)
            dest_type = floatType;
        {
            /// const/high16 vx, lit16 : vx = lit16 << 16
            auto value = static_cast<std::int64_t>(instr->get_source() << 16);

            auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadValue>(
                location,
                dest_type,
                value);

            writeLocalVariable(current_basic_block, dest, gen_value);
        }
        break;
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_WIDE_HIGH16:
        if (!dest_type)
            dest_type = doubleType;
        {
            /// const-wide/high16 vx,lit16 : vx = list16 << 48
            auto value = static_cast<std::int64_t>(instr->get_source() << 48);

            auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadValue>(
                location,
                dest_type,
                value);

            writeLocalVariable(current_basic_block, dest, gen_value);
        }
        break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction21h not supported");
        break;
    }
}
