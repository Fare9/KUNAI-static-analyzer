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
            : IRStmnt()
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

        /**
         * @brief Get destination type where value is assigned.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRAssign::get_destination()
        {
            return destination;
        }

        /**
         * @brief Get the source operand from the assignment.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRAssign::get_source()
        {
            return source;
        }

        /**
         * IRCall class
         */

        /**
         * @brief Constructor of IRCall, this expression represents a Call to a function or method.
         * @param callee: function/method called represented as IRExpr (super-super-class of IRCallee).
         * @param args: vector of arguments to the function/method.
         * @param left: left part of the call (used for AST).
         * @param right: right part of the call (used for AST).
         * @return void
         */
        IRCall::IRCall(std::shared_ptr<IRExpr> callee,
                std::vector<std::shared_ptr<IRExpr>> args,
                std::shared_ptr<IRExpr> left,
                std::shared_ptr<IRExpr> right)
                : IRExpr(CALL_EXPR_T, left, right)
        {
            this->callee = callee;
            this->args = args;
        }

        /**
         * @brief Destructor of IRCall, nothing to be done.
         * @return void
         */
        IRCall::~IRCall() {}

        /**
         * @brief Get the callee object from the function/method called.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRCall::get_callee()
        {
            return callee;
        }

        /**
         * @brief Get the arguments from the call.
         * @return std::vector<std::shared_ptr<IRExpr>>
         */
        std::vector<std::shared_ptr<IRExpr>> IRCall::get_args()
        {
            return args;
        }


        /**
         * IRLoad class
         */

        /**
         * @brief Constructor of IRLoad class, this class represent a load from memory (using memory or using register).
         * @param destination: register where the value will be stored.
         * @param source: expression from where the memory will be retrieved.
         * @param size: loaded size.
         * @param left: left part of the load (used for AST).
         * @param right: right part of the load (used for AST).
         * @return void
         */
        IRLoad::IRLoad(std::shared_ptr<IRExpr> destination,
                    std::shared_ptr<IRExpr> source,
                    std::uint32_t size,
                    std::shared_ptr<IRExpr> left, 
                    std::shared_ptr<IRExpr> right)
                    : IRExpr(LOAD_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
            this->size = size;
        }

        /**
         * @brief Destructor of IRLoad, nothing to be done.
         * @return void
         */
        IRLoad::~IRLoad() {}

        /**
         * @brief Get destination register from the load.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRLoad::get_destination()
        {
            return destination;
        }

        /**
         * @brief Get the expression from where value is loaded.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRLoad::get_source()
        {
            return source;
        }

        /**
         * @brief Get the loaded size.
         * @return std::uint32_t
         */
        std::uint32_t IRLoad::get_size()
        {
            return size;
        }

        /**
         * IRStore class
         */

        /**
         * @brief Constructor of IRStore class, this represent an store to memory instruction.
         * @param destination: Expression where value is written to.
         * @param source: register with the value to be stored.
         * @param size: size of the stored value.
         * @param left: left part of the store (used for AST).
         * @param right: right part of the store (used for AST).
         * @return void
         */
        IRStore::IRStore(std::shared_ptr<IRExpr> destination,
                    std::shared_ptr<IRExpr> source,
                    std::uint32_t size,
                    std::shared_ptr<IRExpr> left, 
                    std::shared_ptr<IRExpr> right)
                : IRExpr(STORE_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
            this->size = size;
        }

        /**
         * @brief Destructor of IRStore class, nothing to be done.
         * @return void
         */
        IRStore::~IRStore() {}

        /**
         * @brief Get expression destination where value is going to be written.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRStore::get_destination()
        {
            return destination;
        }

        /**
         * @brief Get source where value is taken from.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRStore::get_source()
        {
            return source;
        }

        /**
         * @brief Get size of value to be written.
         * @return std::uint32_t
         */
        std::uint32_t IRStore::get_size()
        {
            return size;
        }

        /**
         * IRZComp class
         */

        /**
         * @brief Constructor of IRZComp, this is a comparison with zero.
         * @param comp: type of comparison (== or !=).
         * @param reg: register used in the comparison.
         * @param left: left part of the zero comparison (used for AST).
         * @param right: right part of the zero comparison (used for AST).
         * @return void
         */

        IRZComp::IRZComp(zero_comp_t comp,
                    std::shared_ptr<IRExpr> reg,
                    std::shared_ptr<IRExpr> left, 
                    std::shared_ptr<IRExpr> right)
                : IRExpr(ZCOMP_EXPR_T, left, right)
        {
            this->comp = comp;
            this->reg = reg;
        }

        /**
         * @brief Destructor of IRZComp, nothing to be done.
         * @return void
         */
        IRZComp::~IRZComp() {}

        /**
         * @brief Get register used in the comparison.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRZComp::get_reg()
        {
            return reg;
        }

        /**
         * @brief Get the type of comparison with zero.
         * @return zero_comp_t
         */
        IRZComp::zero_comp_t IRZComp::get_comparison()
        {
            return comp;
        }

        /**
         * IRBComp class
         */

        /**
         * @brief Constructor of IRBComp, this class represent a comparison between two types.
         * @param comp: type of comparison from an enum.
         * @param reg1: first type where the comparison is applied.
         * @param reg2: second type where the comparison is applied.
         * @param left: left part of the comparison (used for AST).
         * @param right: right part of the comparison (used for AST).
         * @return void
         */
        IRBComp::IRBComp(comp_t comp,
                    std::shared_ptr<IRExpr> reg1,
                    std::shared_ptr<IRExpr> reg2,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right)
                : IRExpr(BCOMP_EXPR_T, left, right)
        {
            this->comp = comp;
            this->reg1 = reg1;
            this->reg2 = reg2;
        }

        /**
         * @brief Destructor of IRBComp, nothing to be done.
         * @return void
         */
        IRBComp::~IRBComp() {}

        /**
         * @brief Return the first part of the comparison.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBComp::get_reg1()
        {
            return reg1;
        }

        /**
         * @brief Return the second part of the comparison.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBComp::get_reg2()
        {
            return reg2;
        }

        /**
         * @brief Return the type of comparison
         * @return comp_t
         */
        IRBComp::comp_t IRBComp::get_comparison()
        {
            return comp;
        }
    }
}