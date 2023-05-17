#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

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
