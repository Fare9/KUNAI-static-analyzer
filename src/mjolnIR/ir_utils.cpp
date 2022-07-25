#include "KUNAI/mjolnIR/ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        irfield_t field_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::FIELD_OP_T)
                return std::dynamic_pointer_cast<IRField>(instr);
            return nullptr;
        }

        ircallee_t callee_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::CALLEE_OP_T)
                return std::dynamic_pointer_cast<IRCallee>(instr);
            return nullptr;
        }

        irclass_t class_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::CLASS_OP_T)
                return std::dynamic_pointer_cast<IRClass>(instr);
            return nullptr;
        }

        irstring_t string_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::STRING_OP_T)
                return std::dynamic_pointer_cast<IRString>(instr);
            return nullptr;
        }

        irmemory_t memory_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::MEM_OP_T)
                return std::dynamic_pointer_cast<IRMemory>(instr);
            return nullptr;
        }

        irconstint_t const_int_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::CONST_INT_OP_T)
                return std::dynamic_pointer_cast<IRConstInt>(instr);
            return nullptr;
        }

        irtempreg_t temp_reg_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::TEMP_REGISTER_OP_T)
                return std::dynamic_pointer_cast<IRTempReg>(instr);
            return nullptr;
        }

        irreg_t register_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::REGISTER_OP_T)
                return std::dynamic_pointer_cast<IRReg>(instr);
            return nullptr;
        }

        irujmp_t unconditional_jump_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::UJMP_OP_T)
                return std::dynamic_pointer_cast<IRUJmp>(instr);
            return nullptr;
        }

        ircjmp_t conditional_jump_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::CJMP_OP_T)
                return std::dynamic_pointer_cast<IRCJmp>(instr);
            return nullptr;
        }

        irret_t ret_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::RET_OP_T)
                return std::dynamic_pointer_cast<IRRet>(instr);
            return nullptr;
        }

        ircall_t call_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::CALL_OP_T)
                return std::dynamic_pointer_cast<IRCall>(instr);
            return nullptr;
        }

        irswitch_t switch_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::SWITCH_OP_T)
                return std::dynamic_pointer_cast<IRSwitch>(instr);
            return nullptr;
        }

        irzcomp_t zcomp_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::ZCOMP_OP_T)
                return std::dynamic_pointer_cast<IRZComp>(instr);
            return nullptr;
        }

        irbcomp_t bcomp_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::BCOMP_OP_T)
                return std::dynamic_pointer_cast<IRBComp>(instr);
            return nullptr;
        }

        bool is_cmp(irstmnt_t &instr)
        {
            if (bcomp_ir(instr) == nullptr && zcomp_ir(instr))
                return false;
            return true;
        }

        irnop_t nop_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::NOP_OP_T)
                return std::dynamic_pointer_cast<IRNop>(instr);
            return nullptr;
        }

        irexpr_t expr_ir(irstmnt_t &instr)
        {
            return std::dynamic_pointer_cast<IRExpr>(instr);
        }

        irbinop_t bin_op_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::BINOP_OP_T)
                return std::dynamic_pointer_cast<IRBinOp>(instr);
            return nullptr;
        }

        irunaryop_t unary_op_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::UNARYOP_OP_T)
                return std::dynamic_pointer_cast<IRUnaryOp>(instr);
            return nullptr;
        }

        irassign_t assign_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::ASSIGN_OP_T)
                return std::dynamic_pointer_cast<IRAssign>(instr);
            return nullptr;
        }

        irphi_t phi_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::PHI_OP_T)
                return std::dynamic_pointer_cast<IRPhi>(instr);
            return nullptr;
        }

        irload_t load_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::LOAD_OP_T)
                return std::dynamic_pointer_cast<IRLoad>(instr);
            return nullptr;
        }

        irstore_t store_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::STORE_OP_T)
                return std::dynamic_pointer_cast<IRStore>(instr);
            return nullptr;
        }

        irnew_t new_ir(irstmnt_t &instr)
        {
            if (instr->get_op_type() == IRStmnt::NEW_OP_T)
                return std::dynamic_pointer_cast<IRNew>(instr);
            return nullptr;
        }

        irtype_t type_ir(irstmnt_t &instr)
        {
            return std::dynamic_pointer_cast<IRType>(instr);
        }
    }
}