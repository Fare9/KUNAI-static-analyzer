#include "ir_grammar.hpp"


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
         * @brief Return correct string representation for IRStmnt.
         * 
         * @return std::string 
         */
        std::string IRStmnt::to_string()
        {
            if (stmnt_type == UJMP_STMNT_T)
            {
                auto ujmp = reinterpret_cast<IRUJmp*>(this);

                return ujmp->to_string();
            }
            else if (stmnt_type == CJMP_STMNT_T)
            {
                auto cjmp = reinterpret_cast<IRCJmp*>(this);

                return cjmp->to_string();
            }
            else if (stmnt_type == RET_STMNT_T)
            {
                auto ret = reinterpret_cast<IRRet*>(this);

                return ret->to_string();
            }
            else if (stmnt_type == EXPR_STMNT_T)
            {
                auto expr = reinterpret_cast<IRExpr*>(this);

                return expr->to_string();
            }
            else if (stmnt_type == NONE_STMNT_T)
            {
                return "IRStmnt [NONE]";
            }

            return "";
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
            : IRStmnt(UJMP_STMNT_T)
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
         * @brief Return a string representation of IRUJmp.
         * 
         * @return std::string 
         */
        std::string IRUJmp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRUJmp ";

            str_stream << "[Target: " << target->to_string() << "]";

            return str_stream.str();
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
            : IRStmnt(CJMP_STMNT_T)
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
         * @brief Return string representation of IRCJmp.
         * 
         * @return std::string 
         */
        std::string IRCJmp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRCJmp ";

            str_stream << "[Condition: " << condition->to_string() << "]";
            str_stream << "[Target: " << target->to_string() << "]";
            str_stream << "[Fallthroguh: " << target->to_string() << "]";

            return str_stream.str();
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
            : IRStmnt(RET_STMNT_T)
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
        
        /**
         * @brief Return the string representation of IRRet.
         * 
         * @return std::string
         */
        std::string IRRet::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRRet ";
            str_stream << "[Ret: " << ret_value->to_string() << "]";

            return str_stream.str();
        }
    }
}