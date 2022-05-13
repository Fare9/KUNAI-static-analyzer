#include "optimizer.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        void Optimizer::add_single_stmnt_pass(one_stmnt_opt_t opt)
        {
            // check if pass is already present
            if (std::find(single_statement_optimization.begin(), single_statement_optimization.end(), opt) != std::end(single_statement_optimization))
                return;
            single_statement_optimization.push_back(opt);
        }

        void Optimizer::run_analysis(irgraph_t& func)
        {
            auto& blocks = func->get_nodes();
            for (size_t i = 0, blk_size = blocks.size(); i < blk_size; i++)
            {
                auto& stmnts = blocks[i]->get_statements();

                for (size_t j = 0, stmnt_size = stmnts.size(); j < stmnt_size; j++)
                {
                    irstmnt_t& final_stmnt = stmnts[j];
                    for (auto func : single_statement_optimization)
                        final_stmnt = (*func)(final_stmnt);       
                    stmnts[j] = final_stmnt;             
                }
            }
        }

        irstmnt_t constant_folding(irstmnt_t &instr)
        {
            irstmnt_t assign = nullptr;

            // Check if the instruction is a binary operation
            if (is_bin_op(instr))
            {
                auto bin_op = std::dynamic_pointer_cast<IRBinOp>(instr);

                auto dest = bin_op->get_result();
                auto op1 = std::dynamic_pointer_cast<IRStmnt>(bin_op->get_op1());
                auto op2 = std::dynamic_pointer_cast<IRStmnt>(bin_op->get_op2());
                auto bin_op_type = bin_op->get_bin_op_type();

                if (!is_ir_const_int(op1) || !is_ir_const_int(op2))
                    return instr;

                auto op1_int = std::dynamic_pointer_cast<IRConstInt>(op1);
                auto op2_int = std::dynamic_pointer_cast<IRConstInt>(op2);

                switch (bin_op_type)
                {
                case IRBinOp::ADD_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int + *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::SUB_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int - *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::S_MUL_OP_T:
                case IRBinOp::U_MUL_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int * *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::S_DIV_OP_T:
                case IRBinOp::U_DIV_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int / *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::MOD_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int % *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::AND_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int & *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::OR_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int | *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::XOR_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int ^ *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::SHL_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int << *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRBinOp::SHR_OP_T:
                case IRBinOp::USHR_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int >> *op2_int);

                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                default:
                    return instr;
                }
            }
            else if (is_unary_op(instr))
            {
                auto unary_op = std::dynamic_pointer_cast<IRUnaryOp>(instr);

                auto dest = unary_op->get_result();
                auto op1 = std::dynamic_pointer_cast<IRStmnt>(unary_op->get_op());
                auto unary_op_type = unary_op->get_unary_op_type();

                if (!is_ir_const_int(op1))
                    return instr;

                auto op1_int = std::dynamic_pointer_cast<IRConstInt>(op1);

                switch (unary_op_type)
                {
                case IRUnaryOp::INC_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>((*op1_int)++);
                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRUnaryOp::DEC_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>((*op1_int)--);
                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRUnaryOp::NOT_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(!(*op1_int));
                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                case IRUnaryOp::NEG_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(~(*op1_int));
                    assign = std::make_shared<IRAssign>(dest, const_int, nullptr, nullptr);
                    break;
                }
                default:
                    return instr;
                }
            }
            else
                return instr;

            return assign;
        }

        void Optimizer::fallthrough_target_analysis(MJOLNIR::irgraph_t &ir_graph)
        {
            auto nodes = ir_graph->get_nodes();

            for (auto node : nodes)
            {

                if (node->get_number_of_statements() == 0) // security check
                    continue;

                auto last_inst = node->get_statements().back();

                if (MJOLNIR::is_conditional_jump(last_inst) || MJOLNIR::is_unconditional_jump(last_inst) || MJOLNIR::is_ret(last_inst))
                    continue;

                for (auto aux : nodes)
                {
                    if (node->get_end_idx() == aux->get_start_idx())
                    {
                        ir_graph->add_uniq_edge(node, aux);
                    }
                }
            }
        }

        optimizer_t NewDefaultOptimizer()
        {
            auto optimizer = std::make_shared<Optimizer>();

            optimizer->add_single_stmnt_pass(constant_folding);

            return optimizer;
        }
    }
}