#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction31t *instr)
{
    auto op_code = instr->get_instruction_opcode();
    /// current idx
    auto curr_idx = instr->get_address();

    auto location = mlir::FileLineColLoc::get(&context, module_name, curr_idx, 0);

    auto & bbs = current_method->get_basic_blocks();

    /// get checked value
    auto checked_reg = instr->get_ref_register();
    auto checked_reg_value = readLocalVariable(current_basic_block, bbs, checked_reg);

    /// values for generating switch
    mlir::SmallVector<mlir::Block *> caseSuccessors;
    mlir::SmallVector<int32_t> caseValues;

    switch(op_code)
    {
    case KUNAI::DEX::TYPES::OP_PACKED_SWITCH:
    {
        auto packed_switch = instr->get_packed_switch();

        auto default_block = map_blocks[bbs.get_basic_block_by_idx(curr_idx + instr->get_instruction_length())];

        /// set the target blocks
        for (auto target : packed_switch->get_targets())
        {
            auto bb = bbs.get_basic_block_by_idx(curr_idx + target * 2);
            caseSuccessors.push_back(map_blocks[bb]);
        }
        /// set the target block cases
        auto first = packed_switch->get_first_key();
        for (size_t I = 0, E = caseSuccessors.size(); I < E; ++I)
            caseValues.push_back(first+I);

        mlir::SmallVector<mlir::ValueRange> caseOperands(caseSuccessors.size(), {});

        builder.create<::mlir::cf::SwitchOp>(
            location,
            checked_reg_value,
            default_block,
            mlir::ValueRange(),
            builder.getDenseI32ArrayAttr(caseValues),
            caseSuccessors, 
            caseOperands
        );
    }
    break;
    case KUNAI::DEX::TYPES::OP_SPARSE_SWITCH:
    {
        auto sparse_switch = instr->get_sparse_switch();

        auto default_block = map_blocks[bbs.get_basic_block_by_idx(curr_idx + instr->get_instruction_length())];

        for (auto [key, target] : sparse_switch->get_keys_targets())
        {
            auto bb = bbs.get_basic_block_by_idx(curr_idx + target * 2);
            caseSuccessors.push_back(map_blocks[bb]);
            caseValues.push_back(key);
        }

        mlir::SmallVector<mlir::ValueRange> caseOperands(caseSuccessors.size(), {});

        builder.create<::mlir::cf::SwitchOp>(
            location,
            checked_reg_value,
            default_block,
            mlir::ValueRange(),
            builder.getDenseI32ArrayAttr(caseValues),
            caseSuccessors, 
            caseOperands
        );
    }
    }
}