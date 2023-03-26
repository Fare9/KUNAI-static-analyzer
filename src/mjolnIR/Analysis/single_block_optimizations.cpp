#include "KUNAI/mjolnIR/Analysis/single_block_optimizations.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        std::optional<irblock_t> nop_removal(irblock_t &block, irgraph_t &graph)
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

        std::optional<irblock_t> expression_simplifier(irblock_t &block, irgraph_t &graph)
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
                for (size_t i = 0, stmnts_size = stmnts.size(); i < stmnts_size;)
                {
                    // SimplifySubInst
                    // X - (X - Y) -> Y
                    if (bin_op_ir(stmnts[i]) && bin_op_ir(stmnts[i])->get_bin_op_type() == IRBinOp::SUB_OP_T && (stmnts_size - i) >= 2 &&
                        bin_op_ir(stmnts[i + 1]) && bin_op_ir(stmnts[i + 1])->get_bin_op_type() == IRBinOp::SUB_OP_T)
                    {
                        auto first_instr = bin_op_ir(stmnts[i]);
                        auto second_instr = bin_op_ir(stmnts[i + 1]);

                        if (first_instr->get_op1()->equals(first_instr->get_op2()) &&
                            second_instr->get_op1()->equals(first_instr->get_result()))
                        {
                            irassign_t const assign_inst = std::make_shared<IRAssign>(second_instr->get_result(), second_instr->get_op2());
                            new_statements.push_back(assign_inst);
                            i += 2;
                            modified = true;
                            continue;
                        }
                    }

                    // SimplifyAddInst
                    // (X + Y) - X -> Y
                    if (
                        bin_op_ir(stmnts[i]) && bin_op_ir(stmnts[i])->get_bin_op_type() == IRBinOp::ADD_OP_T && 
                        (stmnts_size - i) >= 2
                        && bin_op_ir(stmnts[i + 1]) && bin_op_ir(stmnts[i + 1])->get_bin_op_type() == IRBinOp::SUB_OP_T)
                    {
                        auto first_instr = bin_op_ir(stmnts[i]);
                        auto second_instr = bin_op_ir(stmnts[i + 1]);

                        if (first_instr->get_op1()->equals(second_instr->get_op2()) && first_instr->get_result()->equals(second_instr->get_op1()))
                        {
                            irassign_t const assign_inst = std::make_shared<IRAssign>(second_instr->get_result(), first_instr->get_op2());
                            new_statements.push_back(assign_inst);
                            i += 2;
                            modified = true;
                            continue;
                        }
                    }
                    // or (Y - X) + X -> Y
                    if (
                        bin_op_ir(stmnts[i]) && bin_op_ir(stmnts[i])->get_bin_op_type() == IRBinOp::SUB_OP_T 
                        && (stmnts_size - i) >= 2 
                        && bin_op_ir(stmnts[i + 1]) && bin_op_ir(stmnts[i + 1])->get_bin_op_type() == IRBinOp::ADD_OP_T)
                    {
                        auto first_instr = bin_op_ir(stmnts[i]);
                        auto second_instr = bin_op_ir(stmnts[i + 1]);

                        if (first_instr->get_op2()->equals(second_instr->get_op2()) 
                            && first_instr->get_result()->equals(second_instr->get_op1()))
                        {
                            irassign_t const assign_inst = std::make_shared<IRAssign>(second_instr->get_result(), first_instr->get_op1());
                            new_statements.push_back(assign_inst);
                            i += 2;
                            modified = true;
                            continue;
                        }
                    }

                    // if the instruction has not been optimized,
                    // means it is not an interesting expression
                    // then push it and go ahead.
                    new_statements.push_back(stmnts[i]);
                    i++;
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
    
        std::optional<irblock_t> instruction_combining(irblock_t &block, irgraph_t &graph)
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
                for (size_t i = 0, stmnts_size = stmnts.size(); i < stmnts_size;)
                {
                    // Y = X + <INT>
                    // Z = Y + <INT>
                    // =============
                    // Z = X + (<INT> + <INT>)
                    if (bin_op_ir(stmnts[i]) && bin_op_ir(stmnts[i])->get_bin_op_type() == IRBinOp::ADD_OP_T
                        && (stmnts_size - i) >= 2
                        && bin_op_ir(stmnts[i + 1]) && bin_op_ir(stmnts[i + 1])->get_bin_op_type() == IRBinOp::ADD_OP_T)
                    {
                        auto op1 = bin_op_ir(stmnts[i]);
                        auto op2 = bin_op_ir(stmnts[i+1]);

                        irstmnt_t const_op1 = op1->get_op2();
                        irstmnt_t const_op2 = op2->get_op2();

                        if (op1->get_result()->equals(op2->get_op1())
                            && const_int_ir(const_op1) != nullptr
                            && const_int_ir(const_op2) != nullptr
                            )
                        {
                            auto int1 = const_int_ir(const_op1);
                            auto int2 = const_int_ir(const_op2);

                            auto result = std::make_shared<IRConstInt>(*int1 + *int2);

                            irbinop_t const bin_op = std::make_shared<IRBinOp>(IRBinOp::ADD_OP_T, op2->get_result(), op1->get_op1(), result);
                            new_statements.push_back(bin_op);
                            i += 2;
                            modified = true;
                            continue;
                        }
                    }

                    // (A | (B ^ C)) ^ ((A ^ C) ^ B)
                    // =============================
                    // (A & (B ^ C))
                    if (bin_op_ir(stmnts[i]) && bin_op_ir(stmnts[i])->get_bin_op_type() == IRBinOp::XOR_OP_T
                        && (stmnts_size - i) >= 5
                        && bin_op_ir(stmnts[i+1]) && bin_op_ir(stmnts[i+1])->get_bin_op_type() == IRBinOp::OR_OP_T
                        && bin_op_ir(stmnts[i+2]) && bin_op_ir(stmnts[i+2])->get_bin_op_type() == IRBinOp::XOR_OP_T
                        && bin_op_ir(stmnts[i+3]) && bin_op_ir(stmnts[i+3])->get_bin_op_type() == IRBinOp::XOR_OP_T
                        && bin_op_ir(stmnts[i+4]) && bin_op_ir(stmnts[i+4])->get_bin_op_type() == IRBinOp::XOR_OP_T)
                    {
                        auto first_xor = bin_op_ir(stmnts[i]);
                        auto second_or = bin_op_ir(stmnts[i+1]);
                        auto third_xor = bin_op_ir(stmnts[i+2]);
                        auto fourth_xor = bin_op_ir(stmnts[i+3]);
                        auto fifth_xor = bin_op_ir(stmnts[i+4]);

                        auto B = first_xor->get_op1();
                        auto C = first_xor->get_op2();
                        auto A = second_or->get_op1();

                        if (
                            second_or->get_op2()->equals(first_xor->get_result())
                            && third_xor->get_op1()->equals(A)
                            && third_xor->get_op2()->equals(C)
                            && fourth_xor->get_op1()->equals(third_xor->get_result())
                            && fourth_xor->get_op2()->equals(B)
                            && fifth_xor->get_op1()->equals(second_or->get_result())
                            && fifth_xor->get_op2()->equals(fourth_xor->get_result())
                        )
                        {
                            auto new_temporal = graph->get_last_temporal() + 1;
                            graph->set_last_temporal(new_temporal);
                            
                            std::string const temp_name = "t" + new_temporal;

                            auto temp_reg = std::make_shared<IRTempReg>(new_temporal,temp_name, 4);

                            auto created_xor = std::make_shared<IRBinOp>(IRBinOp::XOR_OP_T, temp_reg, B,C);
                            auto created_and = std::make_shared<IRBinOp>(IRBinOp::AND_OP_T, fifth_xor->get_result(), temp_reg, A);

                            new_statements.push_back(created_xor);
                            new_statements.push_back(created_and);
                            i += 5;
                            modified = true;
                            continue;
                        }
                    }

                    // if the instruction has not been optimized,
                    // means it is not an interesting expression
                    // then push it and go ahead.
                    new_statements.push_back(stmnts[i]);
                    i++;
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