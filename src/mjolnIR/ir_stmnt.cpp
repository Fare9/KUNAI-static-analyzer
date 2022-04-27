#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * IRStmnt class
         */

        IRStmnt::IRStmnt(stmnt_type_t stmnt_type) : stmnt_type(stmnt_type)
        {
        }

        std::string IRStmnt::to_string()
        {
            if (stmnt_type == UJMP_STMNT_T)
            {
                auto ujmp = reinterpret_cast<IRUJmp *>(this);

                return ujmp->to_string();
            }
            else if (stmnt_type == CJMP_STMNT_T)
            {
                auto cjmp = reinterpret_cast<IRCJmp *>(this);

                return cjmp->to_string();
            }
            else if (stmnt_type == RET_STMNT_T)
            {
                auto ret = reinterpret_cast<IRRet *>(this);

                return ret->to_string();
            }
            else if (stmnt_type == NOP_STMNT_T)
            {
                auto nop = reinterpret_cast<IRNop *>(this);

                return nop->to_string();
            }
            else if (stmnt_type == EXPR_STMNT_T)
            {
                auto expr = reinterpret_cast<IRExpr *>(this);

                return expr->to_string();
            }
            else if (stmnt_type == SWITCH_STMNT_T)
            {
                auto switch_ = reinterpret_cast<IRSwitch *>(this);

                return switch_->to_string();
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

        IRUJmp::IRUJmp(uint64_t addr, irblock_t target)
            : IRStmnt(UJMP_STMNT_T),
              addr(addr),
              target(target)
        {
        }

        std::string IRUJmp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRUJmp ";
            str_stream << "[Target: 0x" << std::hex << addr << "]";

            return str_stream.str();
        }

        /**
         * IRCJmp target
         */

        IRCJmp::IRCJmp(uint64_t addr, irstmnt_t condition, irblock_t target, irblock_t fallthrough)
            : IRStmnt(CJMP_STMNT_T),
              addr(addr),
              condition(condition),
              target(target),
              fallthrough(fallthrough)
        {
        }

        std::string IRCJmp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRCJmp ";

            str_stream << "[Condition: " << condition->to_string() << "]";
            str_stream << "[Target: 0x" << std::hex << addr << "]";

            return str_stream.str();
        }

        /**
         * IRRet class
         */

        IRRet::IRRet(irstmnt_t ret_value)
            : IRStmnt(RET_STMNT_T),
              ret_value(ret_value)
        {
        }

        std::string IRRet::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRRet ";

            if (ret_value != nullptr)
                str_stream << "[Ret: " << ret_value->to_string() << "]";

            return str_stream.str();
        }

        /**
         * IRNop Class
         */

        IRNop::IRNop()
            : IRStmnt(NOP_STMNT_T)
        {
        }

        /**
         * IRSwitch class
         */

        IRSwitch::IRSwitch(std::vector<int32_t> offsets,
                           irexpr_t condition,
                           std::vector<int32_t> constants_checks)
            : IRStmnt(SWITCH_STMNT_T),
              offsets(offsets),
              condition(condition),
              constants_checks(constants_checks)
        {
        }

        std::string IRSwitch::to_string()
        {
            std::stringstream stream;

            stream << "IRSwitch ";

            if (condition)
                stream << "[Condition: " << condition->to_string() << "]";

            for (auto check : constants_checks)
                stream << "[Check: " << check << "]";

            for (auto offset : offsets)
                stream << "[Target: 0x" << std::hex << offset << "]";

            return stream.str();
        }
    }
}