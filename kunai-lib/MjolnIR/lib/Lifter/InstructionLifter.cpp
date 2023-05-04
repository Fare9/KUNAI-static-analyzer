//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file InstructionLifter.cpp
#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction *instr)
{
    switch (instr->get_instruction_type())
    {
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION23X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction23x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION12X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction12x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction11x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION10T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction10t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION20T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction20t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION30T:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction30t *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION10X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction10x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11N:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction11n *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21S:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21s *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21H:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21h *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION51L:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction51l *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION35C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction35c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction21c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION32X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction32x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31I:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction31i *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31C:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction31c *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22S:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22s *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22B:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction22b *>(instr));
        break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: InstructionType not implemented");
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction23x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src1 = instr->get_first_source();
    auto src2 = instr->get_second_source();

    mlir::Type dest_type = nullptr;

    switch (op_code)
    {
    /// Different Add Operations
    case KUNAI::DEX::TYPES::OP_ADD_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_ADD_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_ADD_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_ADD_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AddOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    /// Different Sub operations
    case KUNAI::DEX::TYPES::OP_SUB_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SUB_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_SUB_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_SUB_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::SubOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    /// Different Mul operations
    case KUNAI::DEX::TYPES::OP_MUL_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_MUL_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_MUL_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_MUL_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::MulOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    /// Different Div operations
    case KUNAI::DEX::TYPES::OP_DIV_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_DIV_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_DIV_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_DIV_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::DivOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    /// Different Rem operations
    case KUNAI::DEX::TYPES::OP_REM_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_REM_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_REM_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_REM_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::RemOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    /// All And operations
    case KUNAI::DEX::TYPES::OP_AND_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_AND_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AndOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    /// All Or operations
    case KUNAI::DEX::TYPES::OP_OR_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_OR_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::OrOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    /// All Xor operations
    case KUNAI::DEX::TYPES::OP_XOR_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_XOR_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::XorOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    /// All SHL instructions
    case KUNAI::DEX::TYPES::OP_SHL_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SHL_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shl>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    /// All SHR instructions
    case KUNAI::DEX::TYPES::OP_SHR_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SHR_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shr>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    /// All USHR instructions
    case KUNAI::DEX::TYPES::OP_USHR_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_USHR_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src2);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::UShr>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Opcode from Instruction23x not implemented");
        break;
    }
}

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
        auto reg_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);

        builder.create<::mlir::KUNAI::MjolnIR::ReturnOp>(
            location,
            reg_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_MOVE_RESULT:
    case KUNAI::DEX::TYPES::OP_MOVE_RESULT_WIDE:
    case KUNAI::DEX::TYPES::OP_MOVE_RESULT_OBJECT:
    {
        if (auto call = mlir::dyn_cast<::mlir::KUNAI::MjolnIR::InvokeOp>(map_blocks[current_basic_block]->back()))
        {
            if (call.getNumResults() == 0)
                break;
            auto call_result = call.getResult(0);
            writeLocalVariable(current_basic_block, dest, call_result);
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

void Lifter::gen_instruction(KUNAI::DEX::Instruction12x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    mlir::Type dest_type = nullptr;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_MOVE:
    case KUNAI::DEX::TYPES::OP_MOVE_WIDE:
    case KUNAI::DEX::TYPES::OP_MOVE_OBJECT:
    {
        auto src_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::MoveOp>(
            location,
            src_value.getType(),
            src_value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_ADD_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_ADD_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_ADD_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_ADD_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AddOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_SUB_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SUB_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_SUB_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_SUB_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::SubOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_MUL_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_MUL_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_MUL_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_MUL_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::MulOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_DIV_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_DIV_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_DIV_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_DIV_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::DivOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_REM_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_REM_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_REM_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_REM_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::RemOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_AND_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_AND_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AndOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_OR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_OR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::OrOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_XOR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_XOR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::XorOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_SHL_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SHL_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shl>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_SHR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SHR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shr>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_USHR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_USHR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::UShr>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_NEG_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_NEG_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_NEG_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_NEG_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Neg>(
                location,
                dest_type,
                src_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_NOT_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_NOT_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Not>(
                location,
                dest_type,
                src_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    /// casts
    case KUNAI::DEX::TYPES::OP_INT_TO_LONG:
    case KUNAI::DEX::TYPES::OP_FLOAT_TO_LONG:
    case KUNAI::DEX::TYPES::OP_DOUBLE_TO_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_INT_TO_FLOAT:
    case KUNAI::DEX::TYPES::OP_LONG_TO_FLOAT:
    case KUNAI::DEX::TYPES::OP_DOUBLE_TO_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_INT_TO_DOUBLE:
    case KUNAI::DEX::TYPES::OP_LONG_TO_DOUBLE:
    case KUNAI::DEX::TYPES::OP_FLOAT_TO_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
    case KUNAI::DEX::TYPES::OP_LONG_TO_INT:
    case KUNAI::DEX::TYPES::OP_FLOAT_TO_INT:
    case KUNAI::DEX::TYPES::OP_DOUBLE_TO_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_INT_TO_BYTE:
        if (!dest_type)
            dest_type = byteType;
    case KUNAI::DEX::TYPES::OP_INT_TO_CHAR:
        if (!dest_type)
            dest_type = charType;
    case KUNAI::DEX::TYPES::OP_INT_TO_SHORT:
        if (!dest_type)
            dest_type = shortType;
        {
            auto src_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::CastOp>(
                location,
                dest_type,
                src_value);

            writeLocalVariable(current_basic_block, dest, generated_value);
        }
        break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction12x not supported");
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction22c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto reg = instr->get_destination();

    mlir::Type destination_type;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_IGET:
    case KUNAI::DEX::TYPES::OP_IGET_WIDE:
    {
        if (!destination_type)
        {
            auto field = instr->get_checked_field();
            destination_type = get_type(field->get_type());
        }
    }
    case KUNAI::DEX::TYPES::OP_IGET_BOOLEAN:
        if (!destination_type)
            destination_type = boolType;
    case KUNAI::DEX::TYPES::OP_IGET_BYTE:
        if (!destination_type)
            destination_type = byteType;
    case KUNAI::DEX::TYPES::OP_IGET_CHAR:
        if (!destination_type)
            destination_type = charType;
    case KUNAI::DEX::TYPES::OP_IGET_SHORT:
        if (!destination_type)
            destination_type = shortType;
        {
            auto field = instr->get_checked_field();
            auto field_ref = instr->get_checked_id();

            std::string &field_name = field->get_name();
            std::string &field_class = field->get_class()->get_raw();

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::LoadFieldOp>(
                location,
                destination_type,
                field_name,
                field_class,
                field_ref);

            writeLocalVariable(current_basic_block, reg, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_IPUT:
    case KUNAI::DEX::TYPES::OP_IPUT_WIDE:
    case KUNAI::DEX::TYPES::OP_IPUT_BOOLEAN:
    case KUNAI::DEX::TYPES::OP_IPUT_BYTE:
    case KUNAI::DEX::TYPES::OP_IPUT_CHAR:
    case KUNAI::DEX::TYPES::OP_IPUT_SHORT:
    {
        auto field = instr->get_checked_field();
        auto field_ref = instr->get_checked_id();

        std::string &field_name = field->get_name();
        std::string &field_class = field->get_class()->get_raw();

        auto reg_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), reg);

        builder.create<::mlir::KUNAI::MjolnIR::StoreFieldOp>(
            location,
            reg_value,
            field_name,
            field_class,
            field_ref);
    }
    break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction22c not implemented yet");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction22t *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto v1 = instr->get_first_operand();
    auto v2 = instr->get_second_operand();

    mlir::Value cmp_value;

    mlir::Type I1 = ::mlir::IntegerType::get(&context, 1);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_IF_EQ:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpEq>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_NE:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpNEq>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_LT:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpLt>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_GE:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpGe>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_GT:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpGt>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_LE:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpLe>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }

        auto location_jcc = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 1);
        /// get the addresses from the blocks
        auto true_idx = instr->get_address() + (instr->get_offset() * 2);
        auto false_idx = instr->get_address() + instr->get_instruction_length();
        /// get the blocks:
        ///     - current_block: for obtaining the required arguments.
        ///     - true_block: for generating branch to `true` block
        ///     - false_block: for generating fallthrough to `false` block.
        auto true_block = current_method->get_basic_blocks().get_basic_block_by_idx(true_idx);
        auto false_block = current_method->get_basic_blocks().get_basic_block_by_idx(false_idx);
        /// create the conditional branch
        builder.create<::mlir::cf::CondBranchOp>(
            location_jcc,
            cmp_value,
            map_blocks[true_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, true_block)],
            map_blocks[false_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, false_block)]);
    }
    break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Error Instruction22t not supported");
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction21t *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto v1 = instr->get_check_reg();

    auto v2 = builder.create<::mlir::arith::ConstantIntOp>(location, 0, 32);

    mlir::Value cmp_value;

    mlir::Type I1 = ::mlir::IntegerType::get(&context, 1);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_IF_EQZ:
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpEqz>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                v2);
        }
    case KUNAI::DEX::TYPES::OP_IF_NEZ:
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpNeqz>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                v2);
        }
    case KUNAI::DEX::TYPES::OP_IF_LTZ:
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpLtz>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                v2);
        }
    case KUNAI::DEX::TYPES::OP_IF_GEZ:
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpGez>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                v2);
        }
    case KUNAI::DEX::TYPES::OP_IF_GTZ:
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpGtz>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                v2);
        }
    case KUNAI::DEX::TYPES::OP_IF_LEZ:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::KUNAI::MjolnIR::CmpLez>(
                location,
                I1,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                v2);
        }

        auto location_jcc = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 1);
        /// get the addresses from the blocks
        auto true_idx = instr->get_address() + (instr->get_jump_offset() * 2);
        auto false_idx = instr->get_address() + instr->get_instruction_length();
        /// get the blocks:
        ///     - current_block: for obtaining the required arguments.
        ///     - true_block: for generating branch to `true` block
        ///     - false_block: for generating fallthrough to `false` block.
        auto true_block = current_method->get_basic_blocks().get_basic_block_by_idx(true_idx);
        auto false_block = current_method->get_basic_blocks().get_basic_block_by_idx(false_idx);
        /// create the conditional branch
        builder.create<::mlir::cf::CondBranchOp>(
            location_jcc,
            cmp_value,
            map_blocks[true_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, true_block)],
            map_blocks[false_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, false_block)]);
    }

    break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Error Instruction21t not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction10t *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_GOTO:
    {
        auto offset = instr->get_jump_offset();
        auto target_idx = instr->get_address() + (offset * 2);

        auto target_block = current_method->get_basic_blocks().get_basic_block_by_idx(target_idx);

        builder.create<::mlir::cf::BranchOp>(
            location,
            map_blocks[target_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, target_block)]);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction10t not supported or not recognized.");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction20t *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_GOTO_16:
    {
        auto offset = instr->get_offset();
        auto target_idx = instr->get_address() + (offset * 2);

        auto target_block = current_method->get_basic_blocks().get_basic_block_by_idx(target_idx);

        builder.create<::mlir::cf::BranchOp>(
            location,
            map_blocks[target_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, target_block)]);
    }
    break;

    default:
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction30t *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_GOTO_32:
    {
        auto offset = instr->get_offset();
        auto target_idx = instr->get_address() + (offset * 2);

        auto target_block = current_method->get_basic_blocks().get_basic_block_by_idx(target_idx);

        builder.create<::mlir::cf::BranchOp>(
            location,
            map_blocks[target_block],
            CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, target_block)]);
    }
    break;

    default:
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction10x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_RETURN_VOID:
        builder.create<::mlir::KUNAI::MjolnIR::ReturnOp>(
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
            ::mlir::KUNAI::MjolnIR::DVMByteType::get(&context),
            value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction11n not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction21s *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    mlir::Type dest_type;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_16:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::opcodes::OP_CONST_WIDE_16:
        if (!dest_type)
            dest_type = longType;
        {
            auto value = static_cast<std::int64_t>(instr->get_source());

            auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadValue>(
                location,
                dest_type,
                value);
            writeLocalVariable(current_basic_block, dest, gen_value);
        }
        break;
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction21s not supported");
        break;
    }
}

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

void Lifter::gen_instruction(KUNAI::DEX::Instruction51l *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    /// we will take the registers as big enough
    /// for storing a 64-bit value
    auto dest_reg = instr->get_first_register();

    mlir::Type dest_type;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_CONST_WIDE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto value = static_cast<std::int64_t>(instr->get_wide_value());

            auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadValue>(
                location,
                dest_type,
                value);

            writeLocalVariable(current_basic_block, dest_reg, gen_value);
        }
        break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction51l not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction35c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_INVOKE_VIRTUAL:
    case KUNAI::DEX::TYPES::OP_INVOKE_SUPER:
    case KUNAI::DEX::TYPES::OP_INVOKE_DIRECT:
    case KUNAI::DEX::TYPES::OP_INVOKE_STATIC:
    case KUNAI::DEX::TYPES::OP_INVOKE_INTERFACE:
    {
        mlir::SmallVector<mlir::Value, 4> parameters;

        auto called_method = instr->get_method();
        auto method_ref = instr->get_type_idx();
        auto method_name = called_method->get_name();

        auto parameters_protos = called_method->get_proto()->get_parameters();
        auto invoke_parameters = instr->get_registers();

        ::mlir::Type retType = get_type(called_method->get_proto()->get_return_type());

        bool is_static = op_code == KUNAI::DEX::TYPES::OP_INVOKE_STATIC ? true : false;

        for (size_t I = 0, P = 0, Limit = invoke_parameters.size();
             I < Limit;
             ++I)
        {
            parameters.push_back(readLocalVariable(current_basic_block,
                                                   current_method->get_basic_blocks(), invoke_parameters[I]));

            /// If the method is not static, the first
            /// register is a pointer to the object
            if (I == 0 && op_code != KUNAI::DEX::TYPES::OP_INVOKE_STATIC)
                continue;

            auto fundamental = reinterpret_cast<KUNAI::DEX::DVMFundamental *>(parameters_protos[P]);

            /// if the parameter is a long or a double, skip the second register
            if (fundamental &&
                    fundamental->get_fundamental_type() == KUNAI::DEX::DVMFundamental::LONG ||
                fundamental->get_fundamental_type() == KUNAI::DEX::DVMFundamental::DOUBLE)
                ++I; // skip next register
            /// go to next parameter
            P++;
        }

        if (mlir::isa<::mlir::KUNAI::MjolnIR::DVMVoidType>(retType))
        {
            mlir::Type NoneType;
            builder.create<::mlir::KUNAI::MjolnIR::InvokeOp>(
                location,
                NoneType,
                method_name,
                method_ref,
                is_static,
                parameters);
        }
        else
            builder.create<::mlir::KUNAI::MjolnIR::InvokeOp>(
                location,
                retType,
                method_name,
                method_ref,
                is_static,
                parameters);
    }
    /* code */
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction35c not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction21c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_NEW_INSTANCE:
    {
        auto cls = instr->get_source_dvmclass();

        auto cls_type = get_type(cls);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::NewOp>(
            location,
            cls_type);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_CONST_STRING:
    {
        auto str_value = instr->get_source_str();

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadString>(
            location,
            strObjectType,
            str_value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_SGET:
    case KUNAI::DEX::TYPES::OP_SGET_WIDE:
    case KUNAI::DEX::TYPES::OP_SGET_OBJECT:
    case KUNAI::DEX::TYPES::OP_SGET_BOOLEAN:
    case KUNAI::DEX::TYPES::OP_SGET_BYTE:
    case KUNAI::DEX::TYPES::OP_SGET_CHAR:
    case KUNAI::DEX::TYPES::OP_SGET_SHORT:
    {
        auto field = instr->get_source_field();
        auto field_ref = instr->get_source();

        std::string &field_name = field->get_name();
        std::string &field_class = field->get_class()->get_raw();

        auto dest_type = get_type(field->get_type());

        auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::LoadFieldOp>(
            location,
            dest_type,
            field_name,
            field_class,
            field_ref);

        writeLocalVariable(current_basic_block, dest, generated_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction21c not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction22x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_MOVE_FROM16:
    case KUNAI::DEX::TYPES::OP_MOVE_WIDE_FROM16:
    case KUNAI::DEX::TYPES::OP_MOVE_OBJECT_FROM16:
    {
        auto src_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::MoveOp>(
            location,
            src_value.getType(),
            src_value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction22x not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction32x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_MOVE_16:
    case KUNAI::DEX::TYPES::OP_MOVE_WIDE_16:
    case KUNAI::DEX::TYPES::OP_MOVE_OBJECT_16:
    {
        auto src_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::MoveOp>(
            location,
            src_value.getType(),
            src_value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction32x not supported");
        break;
    }
}

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

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadString>(
            location,
            strObjectType,
            str_value);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction31c not supported");
        break;
    }
}

void Lifter::gen_instruction(KUNAI::DEX::Instruction22s *instr)
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
    case KUNAI::DEX::TYPES::OP_ADD_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_SUB_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_MUL_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_DIV_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_REM_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_AND_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_OR_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    case KUNAI::DEX::TYPES::OP_XOR_INT_LIT16:
        if (!val)
            val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 16);
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
    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction22s not supported");
        break;
    }
}

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
