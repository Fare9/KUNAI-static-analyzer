#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

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
            cmp_value = builder.create<::mlir::arith::CmpIOp>(
                location,
                I1,
                ::mlir::arith::CmpIPredicate::eq,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_NE:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::arith::CmpIOp>(
                location,
                I1,
                ::mlir::arith::CmpIPredicate::ne,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_LT:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::arith::CmpIOp>(
                location,
                I1,
                ::mlir::arith::CmpIPredicate::slt,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_GE:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::arith::CmpIOp>(
                location,
                I1,
                ::mlir::arith::CmpIPredicate::sge,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_GT:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::arith::CmpIOp>(
                location,
                I1,
                ::mlir::arith::CmpIPredicate::sgt,
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                readLocalVariable(current_basic_block, current_method->get_basic_blocks(), v2));
        }
    }
    case KUNAI::DEX::TYPES::OP_IF_LE:
    {
        if (!cmp_value)
        {
            cmp_value = builder.create<::mlir::arith::CmpIOp>(
                location,
                I1,
                ::mlir::arith::CmpIPredicate::sle,
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
