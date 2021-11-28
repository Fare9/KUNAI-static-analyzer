#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * @brief Check if the given statement is a IRField.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_field(std::shared_ptr<IRStmnt> instr)
        {
            auto field = std::dynamic_pointer_cast<IRField>(instr);

            if (field == nullptr)
                return false;

            return true;
        }

        /**
         * @brief Check if given statement object is
         *        a callee method/function
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ir_callee(std::shared_ptr<IRStmnt> instr)
        {
            auto callee = std::dynamic_pointer_cast<IRCallee>(instr);

            if (callee == nullptr)
                return false;

            return true;
        }

        /**
         * @brief Check if given statement object is a class.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ir_class(std::shared_ptr<IRStmnt> instr)
        {
            auto class_ = std::dynamic_pointer_cast<IRClass>(instr);

            if (class_ == nullptr)
                return false;
            
            return true;
        }


        /**
         * @brief Check if given statement object is a string.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ir_string(std::shared_ptr<IRStmnt> instr)
        {
            auto str = std::dynamic_pointer_cast<IRString>(instr);

            if (str == nullptr)
                return false;

            return true;
        }


        /**
         * @brief Check if given statement object is a memory.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ir_memory(std::shared_ptr<IRStmnt> instr)
        {
            auto mem = std::dynamic_pointer_cast<IRMemory>(instr);

            if (mem == nullptr)
                return false;
            
            return true;
        }

        /**
         * @brief Check if given statement object is a constant integer. 
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ir_const_int(std::shared_ptr<IRStmnt> instr)
        {
            auto const_int = std::dynamic_pointer_cast<IRConstInt>(instr);

            if (const_int == nullptr)
                return false;
            
            return true;
        }

        /**
         * @brief Check if a given statement object is a temporal register.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ir_temp_reg(std::shared_ptr<IRStmnt> instr)
        {
            auto temp_reg = std::dynamic_pointer_cast<IRTempReg>(instr);

            if (temp_reg == nullptr)
                return false;

            return true;
        }


        /**
         * @brief Check if the given statement object is
         *        a register.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_register(std::shared_ptr<IRStmnt> instr)
        {
            auto reg = std::dynamic_pointer_cast<IRReg>(instr);

            if (reg == nullptr)
                return false;

            return true;
        }

        /**
         * @brief Check if the given statement is an unconditional jump instruction.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_unconditional_jump(std::shared_ptr<IRStmnt> instr)
        {
            auto ujmp = std::dynamic_pointer_cast<IRUJmp>(instr);

            if (ujmp == nullptr)
                return false;
            
            return true;
        }

        /**
         * @brief Check if the given statement is a conditional jump instruction.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_conditional_jump(std::shared_ptr<IRStmnt> instr)
        {
            auto cjmp = std::dynamic_pointer_cast<IRCJmp>(instr);

            if (cjmp == nullptr)
                return false;
            
            return true;
        }
        
        /**
         * @brief Check if given statement is a ret instruction.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_ret(std::shared_ptr<IRStmnt> instr)
        {
            auto ret = std::dynamic_pointer_cast<IRRet>(instr);

            if (ret == nullptr)
                return false;
            
            return true;
        }

        /**
         * @brief Check if given statement is a call instruction.
         * 
         * @param instr 
         * @return true 
         * @return false 
         */
        bool is_call(std::shared_ptr<IRStmnt> instr)
        {
            auto call = std::dynamic_pointer_cast<IRCall>(instr);

            if (call == nullptr)
                return false;

            return true;
        }
    }
}