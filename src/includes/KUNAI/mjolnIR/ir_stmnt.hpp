/**
 * @file ir_stmnt.hpp
 * @author Farenain
 * 
 * @brief Statement instructions are those that will be executed
 *        one after the other, these will change once we reach one
 *        instruction that change the control flow graph of the function
 *        or method, inside of the statements apart from the expressions
 *        we have those instructions that modify some flags or
 *        temporal registers that are used later in comparison in jumps
 *        to know if a jump is taken or not.
 */

#include <iostream>
#include <memory>
#include <vector>

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRStmnt
        {
        public:
            enum stmnt_type_t
            {
                UJMP_STMNT_T,
                CJMP_STMNT_T,
                RET_STMNT_T,
                EXPR_STMNT_T,
                NONE_STMNT_T = 99 // used to finish the chain of statements
            };

            IRStmnt(stmnt_type_t stmnt_type);
            IRStmnt(stmnt_type_t stmnt_type, std::shared_ptr<IRStmnt> next);
            ~IRStmnt();
            
            stmnt_type_t get_statement_type();

            void set_next_stmnt(std::shared_ptr<IRStmnt> next);
            std::shared_ptr<IRStmnt> get_next_stmnt();

        private:
            //! Type of the statement.
            stmnt_type_t stmnt_type;
            //! next statement in the code.
            std::shared_ptr<IRStmnt> next;
        };

        class IRUJmp : IRStmnt
        {
        public:
            IRUJmp(std::shared_ptr<IRStmnt> target);
            ~IRUJmp();

            std::shared_ptr<IRStmnt> get_jump_target();
        private:
            //! target where the jump will fall
            std::shared_ptr<IRStmnt> target;
        };

        class IRCJmp : IRStmnt
        {
        public:
            IRCJmp(std::shared_ptr<IRStmnt> condition, std::shared_ptr<IRStmnt> target, std::shared_ptr<IRStmnt> fallthrough);
            ~IRCJmp();

            std::shared_ptr<IRStmnt> get_condition();
            std::shared_ptr<IRStmnt> get_jump_target();
            std::shared_ptr<IRStmnt> get_fallthrough_target();
        private:
            //! Condition for taking the target jump
            std::shared_ptr<IRStmnt> condition;
            //! target where the jump will fall
            std::shared_ptr<IRStmnt> target;
            //! fallthrough target.
            std::shared_ptr<IRStmnt> fallthrough;
        };

        class IRRet : IRStmnt
        {
        public:
            IRRet(std::shared_ptr<IRStmnt> ret_value);
            ~IRRet();

            std::shared_ptr<IRStmnt> get_return_value();
        private:
            //! Returned value, commonly a NONE IRType, or an IRReg.
            std::shared_ptr<IRStmnt> ret_value;
        };
    }
}