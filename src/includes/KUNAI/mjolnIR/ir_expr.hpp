/**
 * @file ir_expr.hpp
 * @author Farenain
 * 
 * @brief Different instructions which we consider expressions, these
 *        expressions does not modify the control flow of the program
 *        does not affect to logic of program, for example arithmetic-
 *        logic instructions, types are also expressions.
 */

#include <iostream>
#include <memory>
#include <vector>
#include "ir_stmnt.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRExpr : public IRStmnt
        {
        public:
            enum expr_type_t
            {
                BINOP_EXPR_T,
                UNARYOP_EXPR_T,
                ASSIGN_EXPR_T,
                CALL_EXPR_T,
                TYPE_EXPR_T,
                LOAD_EXPR_T,
                STORE_EXPR_T,
                ZCOMP_EXPR_T,
                BCOMP_EXPR_T,
                NONE_EXPR_T = 99 // used to finish the expressions
            };

            IRExpr(expr_type_t type, std::shared_ptr<IRExpr> left, std::shared_ptr<IRExpr> right);
            ~IRExpr();

            void set_left_expr(std::shared_ptr<IRExpr> left);
            void set_right_expr(std::shared_ptr<IRExpr> right);

            std::shared_ptr<IRExpr> get_left_expr();
            std::shared_ptr<IRExpr> get_right_expr();

            expr_type_t get_expression_type();

            friend bool operator==(IRExpr&, IRExpr&);

        private:
            //! pointers used for the abstract syntax tree
            //! these pointers are common in the IRExpr.
            std::shared_ptr<IRExpr> left;
            std::shared_ptr<IRExpr> right;

            //! ir expression as string
            std::string ir_expr_str;

            //! expression type
            expr_type_t type;
        };

        class IRBinOp : public IRExpr
        {
        public:
            enum bin_op_t
            {
                // common arithmetic instructions
                ADD_OP_T,
                SUB_OP_T,
                S_MUL_OP_T,
                U_MUL_OP_T,
                S_DIV_OP_T,
                U_DIV_OP_T,
                MOD_OP_T,
                // common logic instructions
                AND_OP_T,
                XOR_OP_T,
                OR_OP_T,
                SHL_OP_T,
                SHR_OP_T
            };

            IRBinOp(bin_op_t bin_op_type,
                    std::shared_ptr<IRExpr> result,
                    std::shared_ptr<IRExpr> op1,
                    std::shared_ptr<IRExpr> op2,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);

            ~IRBinOp();

            bin_op_t get_bin_op_type();
            std::shared_ptr<IRExpr> get_result();
            std::shared_ptr<IRExpr> get_op1();
            std::shared_ptr<IRExpr> get_op2();

            friend bool operator==(IRBinOp&, IRBinOp&);
        private:
            //! type of binary operation
            bin_op_t bin_op_type;
            //! IRBinOp =>  IRExpr(result) = IRExpr(op1) <binop> IRExpr(op2)
            //! for the result we will have an IRExpr too.
            std::shared_ptr<IRExpr> result;
            //! now each one of the operators
            std::shared_ptr<IRExpr> op1;
            std::shared_ptr<IRExpr> op2;
        };

        class IRUnaryOp : public IRExpr
        {
        public:
            enum unary_op_t
            {
                INC_OP_T,
                DEC_OP_T,
                NOT_OP_T,
                NEG_OP_T,
                CAST_OP_T,  // maybe not used in binary
                Z_EXT_OP_T, // zero extend
                S_EXT_OP_T  // sign extend
            };

            IRUnaryOp(unary_op_t unary_op_type,
                      std::shared_ptr<IRExpr> result,
                      std::shared_ptr<IRExpr> op,
                      std::shared_ptr<IRExpr> left,
                      std::shared_ptr<IRExpr> right);

            ~IRUnaryOp();

            unary_op_t get_unary_op_type();
            std::shared_ptr<IRExpr> get_result();
            std::shared_ptr<IRExpr> get_op();
            friend bool operator==(IRUnaryOp&, IRUnaryOp&);
        private:
            //! type of unary operation =D
            unary_op_t unary_op_type;
            //! IRUnaryOp => IRExpr(result) = <unaryop> IRExpr(op)
            //! an IRExpr for where the result is stored.
            std::shared_ptr<IRExpr> result;
            // operator
            std::shared_ptr<IRExpr> op;
        };

        class IRAssign : public IRExpr
        {
        public:
            IRAssign(std::shared_ptr<IRExpr> destination,
                     std::shared_ptr<IRExpr> source,
                     std::shared_ptr<IRExpr> left,
                     std::shared_ptr<IRExpr> right);
            ~IRAssign();

            std::shared_ptr<IRExpr> get_destination();
            std::shared_ptr<IRExpr> get_source();
            friend bool operator==(IRAssign&, IRAssign&);
        private:
            //! destination where the value will be stored.
            std::shared_ptr<IRExpr> destination;
            //! source expression from where the value is taken
            std::shared_ptr<IRExpr> source;
        };

        class IRCall : public IRExpr
        {
        public:
            IRCall(std::shared_ptr<IRExpr> callee,
                   std::vector<std::shared_ptr<IRExpr>> args,
                   std::shared_ptr<IRExpr> left,
                   std::shared_ptr<IRExpr> right);

            ~IRCall();

            std::shared_ptr<IRExpr> get_callee();
            std::vector<std::shared_ptr<IRExpr>> get_args();
            friend bool operator==(IRCall&, IRCall&);
        private:
            //! Type representing the function/method called
            std::shared_ptr<IRExpr> callee;
            //! Vector with possible arguments
            std::vector<std::shared_ptr<IRExpr>> args;
        };

        class IRLoad : public IRExpr
        {
        public:
            IRLoad(std::shared_ptr<IRExpr> destination,
                   std::shared_ptr<IRExpr> source,
                   std::uint32_t size,
                   std::shared_ptr<IRExpr> left,
                   std::shared_ptr<IRExpr> right);
            ~IRLoad();

            std::shared_ptr<IRExpr> get_destination();
            std::shared_ptr<IRExpr> get_source();
            std::uint32_t get_size();
            friend bool operator==(IRLoad&, IRLoad&);
        private:
            //! Register where the memory pointed by a register will be loaded.
            std::shared_ptr<IRExpr> destination;
            //! Expression from where memory is read.
            std::shared_ptr<IRExpr> source;
            //! Size of loaded value
            std::uint32_t size;
        };

        class IRStore : public IRExpr
        {
        public:
            IRStore(std::shared_ptr<IRExpr> destination,
                    std::shared_ptr<IRExpr> source,
                    std::uint32_t size,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            ~IRStore();

            std::shared_ptr<IRExpr> get_destination();
            std::shared_ptr<IRExpr> get_source();
            std::uint32_t get_size();
            friend bool operator==(IRStore&, IRStore&);
        private:
            //! Memory pointed by register where value will be stored.
            std::shared_ptr<IRExpr> destination;
            //! Expression with source of value to be stored.
            std::shared_ptr<IRExpr> source;
            //! Size of stored value
            std::uint32_t size;
        };

        class IRZComp : public IRExpr
        {
        public:
            enum zero_comp_t
            {
                EQUAL_ZERO_T,    // ==
                NOT_EQUAL_ZERO_T // !=
            };

            IRZComp(zero_comp_t comp,
                    std::shared_ptr<IRExpr> reg,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            ~IRZComp();

            std::shared_ptr<IRExpr> get_reg();
            zero_comp_t get_comparison();
            friend bool operator==(IRZComp&, IRZComp&);
        private:
            //! Register for comparison with zero.
            std::shared_ptr<IRExpr> reg;
            //! Type of comparison
            zero_comp_t comp;
        };

        class IRBComp : public IRExpr
        {
        public:
            enum comp_t
            {
                EQUAL_T, // ==
                NOT_EQUAL_T, // !=
                GREATER_T, // >
                GREATER_EQUAL_T, // >=
                LOWER_T, // <
                ABOVE_T, // (unsigned) >
                ABOVE_EQUAL_T, // (unsigned) >=
                BELOW_T, // (unsigned) <
            };

            IRBComp(comp_t comp,
                    std::shared_ptr<IRExpr> reg1,
                    std::shared_ptr<IRExpr> reg2,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            ~IRBComp();

            std::shared_ptr<IRExpr> get_reg1();
            std::shared_ptr<IRExpr> get_reg2();
            comp_t get_comparison();
            friend bool operator==(IRBComp&, IRBComp&);
        private:
            //! registers used in the comparisons.
            std::shared_ptr<IRExpr> reg1;
            std::shared_ptr<IRExpr> reg2;
            //! Type of comparison
            comp_t comp;
        };
    }
}