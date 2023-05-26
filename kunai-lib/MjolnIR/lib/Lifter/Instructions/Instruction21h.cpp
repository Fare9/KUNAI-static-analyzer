#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction21h *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_HIGH16:
    {
        auto value = instr->get_source();
        
        union 
        {
            float f;
            std::uint32_t i;
        } conv;
        
        conv.i = value;

        auto gen_value = builder.create<::mlir::arith::ConstantFloatOp>(
            location,
            ::mlir::APFloat(conv.f),
            ::mlir::Float32Type::get(&context));

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_WIDE_HIGH16:
    {
        /// const-wide/high16 vx,lit16 : vx = list16 << 48
        auto value = static_cast<std::int64_t>(instr->get_source() << 48);

        auto gen_value = builder.create<::mlir::arith::ConstantIntOp>(
            location,
            value,
            64);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction21h not supported");
        break;
    }
}
