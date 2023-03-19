#include "KUNAI/mjolnIR/Analysis/single_instruction_optimizations.hpp"

namespace KUNAI 
{
    namespace MJOLNIR
    {

        std::optional<irstmnt_t> constant_folding(irstmnt_t &instr)
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
                        return std::nullopt;
                    }

                    return assign;
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

                return std::nullopt;
            }
            else if (auto unary_op = unary_op_ir(instr))
            {
                auto dest = unary_op->get_result();
                auto op1 = std::dynamic_pointer_cast<IRStmnt>(unary_op->get_op());
                auto unary_op_type = unary_op->get_unary_op_type();

                auto op1_int = std::dynamic_pointer_cast<IRConstInt>(op1);

                if (!op1_int)
                    return std::nullopt;

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
                    return std::nullopt;
                }
            }
            else
                return std::nullopt;

            return assign;
        }

    }
}