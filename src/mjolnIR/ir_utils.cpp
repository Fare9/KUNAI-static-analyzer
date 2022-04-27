#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        
        bool is_ir_field(irstmnt_t instr)
        {
            auto field = std::dynamic_pointer_cast<IRField>(instr);

            if (field == nullptr)
                return false;

            return true;
        }

        
        bool is_ir_callee(irstmnt_t instr)
        {
            auto callee = std::dynamic_pointer_cast<IRCallee>(instr);

            if (callee == nullptr)
                return false;

            return true;
        }

        
        bool is_ir_class(irstmnt_t instr)
        {
            auto class_ = std::dynamic_pointer_cast<IRClass>(instr);

            if (class_ == nullptr)
                return false;
            
            return true;
        }


        bool is_ir_string(irstmnt_t instr)
        {
            auto str = std::dynamic_pointer_cast<IRString>(instr);

            if (str == nullptr)
                return false;

            return true;
        }


        bool is_ir_memory(irstmnt_t instr)
        {
            auto mem = std::dynamic_pointer_cast<IRMemory>(instr);

            if (mem == nullptr)
                return false;
            
            return true;
        }

        
        bool is_ir_const_int(irstmnt_t instr)
        {
            auto const_int = std::dynamic_pointer_cast<IRConstInt>(instr);

            if (const_int == nullptr)
                return false;
            
            return true;
        }

        
        bool is_ir_temp_reg(irstmnt_t instr)
        {
            auto temp_reg = std::dynamic_pointer_cast<IRTempReg>(instr);

            if (temp_reg == nullptr)
                return false;

            return true;
        }


        bool is_register(irstmnt_t instr)
        {
            auto reg = std::dynamic_pointer_cast<IRReg>(instr);

            if (reg == nullptr)
                return false;

            return true;
        }

        
        bool is_unconditional_jump(irstmnt_t instr)
        {
            auto ujmp = std::dynamic_pointer_cast<IRUJmp>(instr);

            if (ujmp == nullptr)
                return false;
            
            return true;
        }

        
        bool is_conditional_jump(irstmnt_t instr)
        {
            auto cjmp = std::dynamic_pointer_cast<IRCJmp>(instr);

            if (cjmp == nullptr)
                return false;
            
            return true;
        }
        
        
        bool is_ret(irstmnt_t instr)
        {
            auto ret = std::dynamic_pointer_cast<IRRet>(instr);

            if (ret == nullptr)
                return false;
            
            return true;
        }

        
        bool is_call(irstmnt_t instr)
        {
            auto call = std::dynamic_pointer_cast<IRCall>(instr);

            if (call == nullptr)
                return false;

            return true;
        }

        
        bool is_switch(irstmnt_t instr)
        {
            auto switch_instr = std::dynamic_pointer_cast<IRSwitch>(instr);

            if (switch_instr == nullptr)
                return false;

            return true;
        }

        bool is_cmp(irstmnt_t instr)
        {
            auto cmp_instr = std::dynamic_pointer_cast<IRBComp>(instr);

            if (cmp_instr != nullptr)
                return true;
            
            auto zcmp_instr = std::dynamic_pointer_cast<IRZComp>(instr);

            if (zcmp_instr != nullptr)
                return true;
            
            return false;
        }

        
    }
}