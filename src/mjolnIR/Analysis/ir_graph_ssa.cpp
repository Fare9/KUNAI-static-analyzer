#include "KUNAI/mjolnIR/Analysis/ir_graph_ssa.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        irstmnt_t IRGraphSSA::translate_instruction(irstmnt_t &instr)
        {
            irstmnt_t new_instr;

            // A = B
            if (auto assign_instr = assign_ir(instr))
            {
                irstmnt_t destination = assign_instr->get_destination();
                irstmnt_t source = assign_instr->get_source();

                if (auto reg = register_ir(destination))
                    destination = rename(reg);

                if (auto reg = register_ir(source))
                    source = reg_last_index.at(reg);

                new_instr = std::make_shared<IRAssign>(std::dynamic_pointer_cast<IRExpr>(destination), std::dynamic_pointer_cast<IRExpr>(source));
            }
            // A = IRUnaryOp B
            else if (auto unary_instr = unary_op_ir(instr))
            {
                irstmnt_t result = unary_instr->get_result();
                irstmnt_t op = unary_instr->get_op();

                if (auto reg = register_ir(result))
                    result = rename(reg);

                if (auto reg = register_ir(op))
                    op = reg_last_index.at(reg);

                if (unary_instr->get_unary_op_type() == IRUnaryOp::CAST_OP_T && unary_instr->get_cast_type() == IRUnaryOp::TO_CLASS)
                    new_instr = std::make_shared<IRUnaryOp>(IRUnaryOp::CAST_OP_T, IRUnaryOp::TO_CLASS, unary_instr->get_class_cast(), result, op);
                else if (unary_instr->get_unary_op_type() == IRUnaryOp::CAST_OP_T)
                    new_instr = std::make_shared<IRUnaryOp>(IRUnaryOp::CAST_OP_T, unary_instr->get_cast_type(), result, op);
                else
                    new_instr = std::make_shared<IRUnaryOp>(unary_instr->get_unary_op_type(), result, op);
            }
            // A = B IRBinaryOp C
            else if (auto binary_instr = bin_op_ir(instr))
            {
                irstmnt_t result = binary_instr->get_result();
                irstmnt_t op1 = binary_instr->get_op1();
                irstmnt_t op2 = binary_instr->get_op2();

                if (auto reg = register_ir(result))
                    result = rename(reg);
                
                if (auto reg = register_ir(op1))
                    op1 = reg_last_index.at(reg);
                
                if (auto reg = register_ir(op2))
                    op2 = reg_last_index.at(reg);

                new_instr = std::make_shared<IRBinOp>(binary_instr->get_bin_op_type(), result, op1, op2);
            }
            else if (auto load_instr = load_ir(instr))
            {
                irstmnt_t destination = load_instr->get_destination();
                irstmnt_t source = load_instr->get_source();
                irstmnt_t index = load_instr->get_index();

                if (auto reg = register_ir(destination))
                    destination = rename(reg);

                if (auto reg = register_ir(source))
                    source = reg_last_index.at(reg);
                
                if (auto reg = register_ir(index))
                    index = reg_last_index.at(reg);

                new_instr = std::make_shared<IRLoad>(destination, source, index, load_instr->get_size());
            }

            return new_instr;
        }

        irreg_t IRGraphSSA::rename(irreg_t old_reg)
        {
            irreg_t new_reg;

            if (reg_last_index.find(old_reg) == reg_last_index.end())
            {
                new_reg = std::make_shared<IRReg>(old_reg->get_id(),
                                                  0,
                                                  old_reg->get_current_arch(),
                                                  "v" + std::to_string(old_reg->get_id()) + "." + std::to_string(0),
                                                  old_reg->get_type_size());
                reg_last_index[old_reg] = new_reg;
            }
            else
            {
                auto current_reg = reg_last_index[old_reg];
                auto new_sub_id = current_reg->get_sub_id() + 1;
                new_reg = std::make_shared<IRReg>(
                    current_reg->get_id(),
                    new_sub_id,
                    current_reg->get_current_arch(),
                    "v" + std::to_string(current_reg->get_id()) + "." + std::to_string(new_sub_id),
                    current_reg->get_type_size());
            }

            return new_reg;
        }

    }
}