#include "ir_stmnt.hpp"


namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * IRStmnt class
         */

        /**
         * @brief Constructor of IRStmnt.
         * @param stmnt_type: type of the statement.
         * @return void
         */
        IRStmnt::IRStmnt(stmnt_type_t stmnt_type) 
        {
            this->stmnt_type = stmnt_type;
        }

        /**
         * @brief Constructor of IRStmnt, assign the next pointer.
         * @param stmnt_type: type of the statement.
         * @param next: pointer to a next statement.
         * @return void
         */
        IRStmnt::IRStmnt(stmnt_type_t stmnt_type, std::shared_ptr<IRStmnt> next)
        {
            this->stmnt_type = stmnt_type;
            this->next = next;
        }

        /**
         * @brief Destructor of IRStmnt, nothing to be done.
         * @return void
         */
        IRStmnt::~IRStmnt() {}

        /**
         * @brief Get the type of the statement.
         * @return stmnt_type_t
         */
        IRStmnt::stmnt_type_t IRStmnt::get_statement_type()
        {
            return stmnt_type;
        }

        /**
         * @brief Set the next statement in the queue to analyze.
         * @param next: next statement for the analysis.
         * @return void
         */
        void IRStmnt::set_next_stmnt(std::shared_ptr<IRStmnt> next)
        {
            this->next = next;
        }

        /**
         * @brief Get the next statement in the queue to analyze.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRStmnt::get_next_stmnt()
        {
            return next;
        }

        /**
         * IRUJmp class
         */
        
        /**
         * @brief Constructor of IRUJmp, this kind of class is an unconditional jump with just one target.
         * @param target: target of the jump.
         * @return void
         */
        IRUJmp::IRUJmp(std::shared_ptr<IRStmnt> target)
            : IRStmnt(UJMP_STMNT_T, std::make_shared<IRStmnt>(NONE_STMNT_T))
        {
            this->target = target;
        }

        /**
         * @brief Destructor of IRUJmp, nothing to be done here.
         * @return void
         */
        IRUJmp::~IRUJmp() {}
        
        /**
         * @brief Get the target of the unconditional jump.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRUJmp::get_jump_target()
        {
            return target;
        }

        /**
         * IRCJmp target
         */
        
        /**
         * @brief Constructor of IRCJmp, this represent a conditional jump.
         * @param condition: condition (commonly a register from a CMP) that if true takes target.
         * @param target: target of the conditional jump.
         * @param fallthrough: address of the statement if the condition is not true.
         * @return void
         */
        IRCJmp::IRCJmp(std::shared_ptr<IRStmnt> condition, std::shared_ptr<IRStmnt> target, std::shared_ptr<IRStmnt> fallthrough)
            : IRStmnt(CJMP_STMNT_T, std::make_shared<IRStmnt>(NONE_STMNT_T))
        {
            this->condition = condition;
            this->target = target;
            this->fallthrough = fallthrough;
        }

        /**
         * @brief Destructor of IRCJmp, nothing to be done.
         * @return void
         */
        IRCJmp::~IRCJmp() {}

        /**
         * @brief Get the condition of the conditional jump.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRCJmp::get_condition()
        {
            return condition;
        }

        /**
         * @brief Get the jump target if the condition is true.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRCJmp::get_jump_target()
        {
            return target;
        }
        
        /**
         * @brief Get the fallthrough target in case condition is false.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRCJmp::get_fallthrough_target()
        {
            return fallthrough;
        }

        /**
         * IRRet class
         */

        /**
         * @brief Constructor of IRRet, this statement represent a return instruction.
         * @param ret_value: return value, this can be a NONE type or a register.
         * @return void
         */
        IRRet::IRRet(std::shared_ptr<IRStmnt> ret_value)
            : IRStmnt(RET_STMNT_T, std::make_shared<IRStmnt>(NONE_STMNT_T))
        {
            this->ret_value = ret_value;
        }

        /**
         * @brief Destructor of IRRet, nothing to be done.
         * @return void
         */
        IRRet::~IRRet() {}

        /**
         * @brief Get the return value this will be a NONE IRType, or an IRReg.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRRet::get_return_value()
        {
            return ret_value;
        }

        
    }
}