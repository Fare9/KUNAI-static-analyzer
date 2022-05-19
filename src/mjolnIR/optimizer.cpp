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

        void Optimizer::run_analysis(irgraph_t &func)
        {
            auto &blocks = func->get_nodes();
            for (size_t i = 0, blk_size = blocks.size(); i < blk_size; i++)
            {
                auto &stmnts = blocks[i]->get_statements();

                for (size_t j = 0, stmnt_size = stmnts.size(); j < stmnt_size; j++)
                {
                    irstmnt_t &final_stmnt = stmnts[j];
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
            if (auto bin_op = bin_op_ir(instr))
            {
                auto dest = bin_op->get_result();
                auto op1 = std::dynamic_pointer_cast<IRStmnt>(bin_op->get_op1());
                auto op2 = std::dynamic_pointer_cast<IRStmnt>(bin_op->get_op2());
                auto bin_op_type = bin_op->get_bin_op_type();


                // REG = X ADD|SUB|MUL|DIV|MOD|AND|OR|XOR Y --> REG = Const
                auto op1_int = const_int_ir(op1);
                auto op2_int = const_int_ir(op2);

                if (op1_int && op2_int)
                {

                    switch (bin_op_type)
                    {
                    case IRBinOp::ADD_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int + *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::SUB_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int - *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::S_MUL_OP_T:
                    case IRBinOp::U_MUL_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int * *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::S_DIV_OP_T:
                    case IRBinOp::U_DIV_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int / *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::MOD_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int % *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::AND_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int & *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::OR_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int | *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::XOR_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int ^ *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::SHL_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int << *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    case IRBinOp::SHR_OP_T:
                    case IRBinOp::USHR_OP_T:
                    {
                        irconstint_t const_int = std::make_shared<IRConstInt>(*op1_int >> *op2_int);

                        assign = std::make_shared<IRAssign>(dest, const_int);
                        break;
                    }
                    default:
                        return instr;
                    }
                }

                if (bin_op_type == IRBinOp::ADD_OP_T)
                {
                    // REG = A + 0 --> REG = A                    
                    if (op2_int && op2_int->get_value_unsigned() == 0)
                    {
                        assign = std::make_shared<IRAssign>(dest, std::dynamic_pointer_cast<IRExpr>(op1));
                        return assign;
                    }
                    // REG = 0 + A --> REG = A
                    if (op1_int && op1_int->get_value_unsigned() == 0)
                    {
                        assign = std::make_shared<IRAssign>(dest, std::dynamic_pointer_cast<IRExpr>(op2));
                        return assign;
                    }
                }

                if (bin_op_type == IRBinOp::S_MUL_OP_T || bin_op_type == IRBinOp::U_MUL_OP_T)
                {
                    // REG = A * 1 --> REG = A
                    if (op2_int && op2_int->get_value_unsigned() == 1)
                    {
                        assign = std::make_shared<IRAssign>(dest, std::dynamic_pointer_cast<IRExpr>(op1));
                        return assign;
                    }

                    // REG = 1 * A --> REG = A
                    if (op1_int && op1_int->get_value_unsigned() == 1)
                    {
                        assign = std::make_shared<IRAssign>(dest, std::dynamic_pointer_cast<IRExpr>(op2));
                        return assign;
                    }
                }

                if (bin_op_type == IRBinOp::SUB_OP_T)
                {
                    // REG = A - 0 --> REG = A
                    if (op2_int && op2_int->get_value_unsigned() == 0)
                    {
                        assign = std::make_shared<IRAssign>(dest, std::dynamic_pointer_cast<IRExpr>(op1));
                        return assign;
                    }

                    // REG = 0 - A --> REG = -A
                    if (op1_int && op1_int->get_value_unsigned() == 0)
                    {
                        assign = std::make_shared<IRUnaryOp>(IRUnaryOp::NEG_OP_T, dest, std::dynamic_pointer_cast<IRExpr>(op2));
                        return assign;
                    }

                }

                return instr;
            }
            else if (auto unary_op = unary_op_ir(instr))
            {
                auto dest = unary_op->get_result();
                auto op1 = std::dynamic_pointer_cast<IRStmnt>(unary_op->get_op());
                auto unary_op_type = unary_op->get_unary_op_type();

                auto op1_int = std::dynamic_pointer_cast<IRConstInt>(op1);

                if (!op1_int)
                    return instr;

                switch (unary_op_type)
                {
                case IRUnaryOp::INC_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>((*op1_int)++);
                    assign = std::make_shared<IRAssign>(dest, const_int);
                    break;
                }
                case IRUnaryOp::DEC_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>((*op1_int)--);
                    assign = std::make_shared<IRAssign>(dest, const_int);
                    break;
                }
                case IRUnaryOp::NOT_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(!(*op1_int));
                    assign = std::make_shared<IRAssign>(dest, const_int);
                    break;
                }
                case IRUnaryOp::NEG_OP_T:
                {
                    irconstint_t const_int = std::make_shared<IRConstInt>(~(*op1_int));
                    assign = std::make_shared<IRAssign>(dest, const_int);
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

                if (MJOLNIR::conditional_jump_ir(last_inst) || MJOLNIR::unconditional_jump_ir(last_inst) || MJOLNIR::ret_ir(last_inst))
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