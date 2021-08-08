#include "ir_expr.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        /**
         * @brief Constructor of IRExpr this will be the most common type of instruction of the IR.
         * @param type: type of the expression (type of the instruction).
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRExpr::IRExpr(expr_type_t type, std::shared_ptr<IRExpr> left, std::shared_ptr<IRExpr> right)
        {
            this->type = type;
            this->left = left;
            this->right = right;
        }

        /**
         * @brief Destructor of IRExpr, nothing to be done.
         * @return void
         */
        IRExpr::~IRExpr() {}

        /**
         * @brief Set the left expression (for the AST).
         * @param left: left part of expression for the AST.
         */
        void IRExpr::set_left_expr(std::shared_ptr<IRExpr> left)
        {
            this->left = left;
        }

        /**
         * @brief Set the right expression (for the AST).
         * @param left: right part of expression for the AST.
         */
        void IRExpr::set_right_expr(std::shared_ptr<IRExpr> right)
        {
            this->right = right;
        }

        /**
         * @brief Get the left expression (from the AST)
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRExpr::get_left_expr()
        {
            return left;
        }

        /**
         * @brief Get the right expression (from the AST)
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRExpr::get_right_expr()
        {
            return right;
        }

        /**
         * @brief Get the type of current expression.
         * @return expr_type_t
         */
        IRExpr::expr_type_t IRExpr::get_expression_type()
        {
            return type;
        }

        /**
         * @brief Constructor of IRBinOp, this class represent different instructions with two operators and a result.
         * @param bin_op_type: type of binary operation.
         * @param result: where result of operation is stored.
         * @param op1: first operand of the operation.
         * @param op2: second operand of the operation.
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRBinOp::IRBinOp(bin_op_t bin_op_type,
                    std::shared_ptr<IRExpr> result, 
                    std::shared_ptr<IRExpr> op1,
                    std::shared_ptr<IRExpr> op2,
                    std::shared_ptr<IRExpr> left, 
                    std::shared_ptr<IRExpr> right)
            : IRExpr(BINOP_EXPR_T, left, right)
        {
            this->bin_op_type = bin_op_type;
            this->result = result;
            this->op1 = op1;
            this->op2 = op2;
        }

        /**
         * @brief Destructor of IRBinOp.
         * @return void
         */
        IRBinOp::~IRBinOp() {}

        /**
         * @brief Get the type of binary operation applied.
         * @return bin_op_t
         */
        IRBinOp::bin_op_t IRBinOp::get_bin_op_type()
        {
            return bin_op_type;
        }

        /**
         * @brief Get expression of type where result is stored.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBinOp::get_result()
        {
            return result;
        }

        /**
         * @brief Get the operand 1 of the operation.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBinOp::get_op1()
        {
            return op1;
        }

        /**
         * @brief Get the operand 2 of the operation.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBinOp::get_op2()
        {
            return op2;
        }

        /**
         * IRUnaryOp class
         */

        /**
         * @brief Constructor of IRUnaryOp class, this will get different values for the instruction.
         * @param unary_op_type: type of unary operation.
         * @param result: where the operation stores the result.
         * @param op: operand from the instruction.
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRUnaryOp::IRUnaryOp(unary_op_t unary_op_type,
                    std::shared_ptr<IRExpr> result, 
                    std::shared_ptr<IRExpr> op,
                    std::shared_ptr<IRExpr> left, 
                    std::shared_ptr<IRExpr> right)
            : IRExpr(UNARYOP_EXPR_T, left, right)
        {
            this->unary_op_type = unary_op_type;
            this->result = result;
            this->op = op;
        }

        /**
         * @brief Destructor of IRUnaryOp class, nothing to be done.
         * @return void
         */
        IRUnaryOp::~IRUnaryOp() {}

        /**
         * @brief Get the type of unary operation of the instruction.
         * @return unary_op_t
         */
        IRUnaryOp::unary_op_t IRUnaryOp::get_unary_op_type()
        {
            return unary_op_type;
        }

        /**
         * @brief Get the IRExpr where result of the operation would be stored.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRUnaryOp::get_result()
        {
            return result;
        }

        /**
         * @brief Get the IRExpr of the operand of the instruction.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRUnaryOp::get_op()
        {
            return op;
        }

        /**
         * IRAssign class
         */

        /**
         * @brief Constructor of IRAssign, this one will have just those from the basic types,
         *        no more information is needed, the left part and the right part.
         * @param destination: type where the value will be assigned.
         * @param source: type from where the value is taken.
         * @param left: left part of the assignment (also used for AST).
         * @param right: right part of the assignment (also used for AST).
         * @return void
         */
        IRAssign::IRAssign(std::shared_ptr<IRExpr> destination,
                    std::shared_ptr<IRExpr> source,
                    std::shared_ptr<IRExpr> left, 
                    std::shared_ptr<IRExpr> right)
            : IRExpr(ASSIGN_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
        }

        /**
         * @brief Destructor of IRAssign, nothing to be done here.
         * @return void
         */
        IRAssign::~IRAssign() {}
    }
}