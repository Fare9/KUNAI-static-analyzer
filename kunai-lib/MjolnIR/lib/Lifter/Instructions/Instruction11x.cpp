#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction11x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_RETURN:
    case KUNAI::DEX::TYPES::OP_RETURN_WIDE:
    case KUNAI::DEX::TYPES::OP_RETURN_OBJECT:
    {
        auto reg_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);

        builder.create<::mlir::func::ReturnOp>(
            location,
            reg_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_MOVE_RESULT:
    case KUNAI::DEX::TYPES::OP_MOVE_RESULT_WIDE:
    case KUNAI::DEX::TYPES::OP_MOVE_RESULT_OBJECT:
    {
        if (auto call = mlir::dyn_cast<::mlir::KUNAI::MjolnIR::InvokeOp>(map_blocks[analysis_context.current_basic_block]->back()))
        {
            if (call.getNumResults() == 0)
                break;
            auto call_result = call.getResult(0);
            writeLocalVariable(analysis_context.current_basic_block, dest, call_result);
        }
        else
            throw exceptions::LifterException("Lifter::gen_instruction: error lifting OP_MOVE_RESULT*, last instruction is not an invoke...");
    }
    break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction11x not supported");
        break;
    }
}
