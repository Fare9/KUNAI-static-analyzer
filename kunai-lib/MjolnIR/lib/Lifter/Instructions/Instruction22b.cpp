#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction22b *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);
    auto location_1 = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 1);

    auto dest = instr->get_destination();
    auto src1 = instr->get_first_operand();
    auto src2 = instr->get_second_operand();

    mlir::Value val;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_ADD_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AddOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_SUB_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::SubOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_MUL_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::MulOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_DIV_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::DivOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_REM_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::RemOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_AND_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AndOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_OR_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::OrOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_XOR_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::XorOp>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_SHL_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shl>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_SHR_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shr>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_USHR_INT_LIT8:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shr>(
                location_1,
                intType,
                src1_value,
                val);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction22b not supported");
        break;
    }
}
