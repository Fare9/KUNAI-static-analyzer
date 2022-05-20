#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        
        irfield_t field_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRField>(instr);
        }

        
        ircallee_t callee_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRCallee>(instr);
        }

        
        irclass_t class_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRClass>(instr);
        }


        irstring_t string_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRString>(instr);
        }


        irmemory_t memory_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRMemory>(instr);
        }

        
        irconstint_t const_int_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRConstInt>(instr);
        }

        
        irtempreg_t temp_reg_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRTempReg>(instr);
        }


        irreg_t register_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRReg>(instr);
        }

        
        irujmp_t unconditional_jump_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRUJmp>(instr);   
        }

        
        ircjmp_t conditional_jump_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRCJmp>(instr);
        }
        
        
        irret_t ret_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRRet>(instr);
        }

        
        ircall_t call_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRCall>(instr);
        }

        
        irswitch_t switch_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRSwitch>(instr);
        }

        irzcomp_t zcomp_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRZComp>(instr);
        }

        irbcomp_t bcomp_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRBComp>(instr);
        }

        bool is_cmp(irstmnt_t& instr)
        {
            if (bcomp_ir(instr) == nullptr && zcomp_ir(instr))
                return false;
            return true;
        }

        irnop_t nop_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRNop>(instr);
        }

        irexpr_t expr_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRExpr>(instr);
        }

        irbinop_t bin_op_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRBinOp>(instr);
        }

        irunaryop_t unary_op_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRUnaryOp>(instr);
        }

        irassign_t assign_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRAssign>(instr);
        }

        irload_t load_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRLoad>(instr);
        }

        irstore_t store_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRStore>(instr);
        }

        irnew_t new_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRNew>(instr);
        }

        irtype_t type_ir(irstmnt_t& instr)
        {
            return std::dynamic_pointer_cast<IRType>(instr);
        }
    }
}