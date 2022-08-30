#include "KUNAI/mjolnIR/Analysis/single_block_optimizations.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        std::optional<irblock_t> nop_removal(irblock_t &block)
        {
            auto &stmnts = block->get_statements();
            bool changed = false;

            for (auto it = stmnts.begin(); it != stmnts.end(); it++)
            {
                auto instr = *it;
                if (nop_ir(instr))
                {
                    it = stmnts.erase(it);
                    changed = true;
                }
            }

            if (!changed)
                return std::nullopt;

            return block;
        }

        std::optional<irblock_t> expression_simplifier(irblock_t &block)
        {
            auto &stmnts = block->get_statements();
            std::vector<KUNAI::MJOLNIR::irstmnt_t> new_statements;

            // The idea here is to create a new vector with new statements
            // also we will apply this optimizations until no more modifications
            // are applied, because we can apply simplification in cascase.
            bool modified = true;
            while (modified)
            {
                // create a new state
                modified = false;
                new_statements.clear();

                // here analyze the instructions, and apply simplifications
                for (size_t index_instruction = 0, stmnts_size = stmnts.size(); index_instruction < stmnts_size;)
                {
                    // X - (X - Y) -> Y
                    if (bin_op_ir(stmnts[index_instruction]) && bin_op_ir(stmnts[index_instruction])->get_bin_op_type() == IRBinOp::SUB_OP_T 
                        && (stmnts_size - index_instruction) >= 2 && 
                        bin_op_ir(stmnts[index_instruction + 1]) && bin_op_ir(stmnts[index_instruction + 1])->get_bin_op_type() == IRBinOp::SUB_OP_T)
                    {
                        auto first_instr = bin_op_ir(stmnts[index_instruction]);
                        auto second_instr = bin_op_ir(stmnts[index_instruction + 1]);

                        if (first_instr->get_op1()->equals(first_instr->get_op2()) &&
                            second_instr->get_op1()->equals(first_instr->get_result()))
                        {
                            irassign_t assign_inst = std::make_shared<IRAssign>(second_instr->get_result(), second_instr->get_op2());
                            new_statements.push_back(assign_inst);
                            index_instruction += 2;
                            modified = true;
                            continue;
                        }
                    }

                    // if the instruction has not been optimized,
                    // means it is not an interesting expression
                    // then push it and go ahead.
                    new_statements.push_back(stmnts[index_instruction]);
                    index_instruction++;
                }

                // update the statements
                if (modified)
                {
                    stmnts.clear();
                    stmnts = new_statements;
                }
            }
            
            return block;
        }
    }
}