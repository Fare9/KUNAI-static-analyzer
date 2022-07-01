#include "KUNAI/mjolnIR/Analysis/ir_graph_ssa.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        IRGraphSSA::IRGraphSSA(irgraph_t &code_graph)
        {
            auto nodes = code_graph->get_nodes();
            auto edges = code_graph->get_edges();

            for (auto node : nodes)
                add_node(node);
            for (auto edge : edges)
                add_edge(edge.first, edge.second);

            collect_var_assign();
            insert_phi_node();
        }

        void IRGraphSSA::collect_var_assign()
        {
            auto &blocks = get_nodes();

            for (auto &block : blocks)
            {
                auto &instrs = block->get_statements();

                for (auto &instr : instrs)
                {
                    // A = B
                    if (auto assign_instr = assign_ir(instr))
                    {
                        irstmnt_t destination = assign_instr->get_destination();

                        if (auto reg = register_ir(destination))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                    // A = IRUnaryOp B
                    else if (auto unary_instr = unary_op_ir(instr))
                    {
                        irstmnt_t result = unary_instr->get_result();

                        if (auto reg = register_ir(result))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                    // A = B IRBinaryOp C
                    else if (auto binary_instr = bin_op_ir(instr))
                    {
                        irstmnt_t result = binary_instr->get_result();

                        if (auto reg = register_ir(result))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                    // A = LOAD(B[INDEX])
                    else if (auto load_instr = load_ir(instr))
                    {
                        irstmnt_t destination = load_instr->get_destination();

                        if (auto reg = register_ir(destination))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                    // A = New Class
                    else if (auto new_ir_instr = new_ir(instr))
                    {
                        irstmnt_t result = new_ir_instr->get_result();

                        if (auto reg = register_ir(result))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                    // call <callee>(arg1, arg2, arg3...)
                    else if (auto call_instr = call_ir(instr))
                    {
                        irstmnt_t ret_val = call_instr->get_ret_val();

                        if (auto reg = register_ir(ret_val))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                    // STORE(A) = B
                    else if (auto store_instr = store_ir(instr))
                    {
                        irstmnt_t destination = store_instr->get_destination();

                        if (auto reg = register_ir(destination))
                        {
                            var_block_map[reg].insert(block);
                            continue;
                        }
                    }
                }
            }
        }

        void IRGraphSSA::insert_phi_node()
        {
            Nodes work_list;
            std::unordered_map<irblock_t, irreg_t> inserted;
            auto dominance_frontier = compute_dominance_frontier();

            for (const auto& p : var_block_map)
            {
                const irreg_t& reg = p.first;
                
                for (auto & block : p.second)
                    work_list.push_back(block);

                while (!work_list.empty())
                {
                    auto& block = work_list.front();
                    work_list.erase(work_list.begin());

                    for (auto & df_block : dominance_frontier.at(block))
                    {
                        if (inserted.at(df_block) != reg)
                        {
                            auto lhs = std::make_shared<IRReg>(*reg.get());
                            auto result = std::make_shared<IRReg>(*reg.get());

                            auto phi_instr = std::make_shared<IRPhi>();
                            phi_instr->add_result(result);
                            phi_instr->add_param(lhs);
                            
                            df_block->add_statement_at_beginning(phi_instr);

                            // finally add the block from dominance_frontier
                            // into the worklist
                            work_list.push_back(df_block);
                        }
                    }
                }
            }
        }

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
                    new_instr = std::make_shared<IRUnaryOp>(IRUnaryOp::CAST_OP_T, IRUnaryOp::TO_CLASS, unary_instr->get_class_cast(), std::dynamic_pointer_cast<IRExpr>(result), std::dynamic_pointer_cast<IRExpr>(op));
                else if (unary_instr->get_unary_op_type() == IRUnaryOp::CAST_OP_T)
                    new_instr = std::make_shared<IRUnaryOp>(IRUnaryOp::CAST_OP_T, unary_instr->get_cast_type(), std::dynamic_pointer_cast<IRExpr>(result), std::dynamic_pointer_cast<IRExpr>(op));
                else
                    new_instr = std::make_shared<IRUnaryOp>(unary_instr->get_unary_op_type(), std::dynamic_pointer_cast<IRExpr>(result), std::dynamic_pointer_cast<IRExpr>(op));
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

                new_instr = std::make_shared<IRBinOp>(binary_instr->get_bin_op_type(), std::dynamic_pointer_cast<IRExpr>(result), std::dynamic_pointer_cast<IRExpr>(op1), std::dynamic_pointer_cast<IRExpr>(op2));
            }
            // A = LOAD(B[INDEX])
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

                new_instr = std::make_shared<IRLoad>(std::dynamic_pointer_cast<IRExpr>(destination), std::dynamic_pointer_cast<IRExpr>(source), std::dynamic_pointer_cast<IRExpr>(index), load_instr->get_size());
            }
            // A = New Class
            else if (auto new_ir_instr = new_ir(instr))
            {
                irstmnt_t result = new_ir_instr->get_result();

                if (auto reg = register_ir(result))
                    result = rename(reg);

                new_instr = std::make_shared<IRNew>(std::dynamic_pointer_cast<IRExpr>(result), new_ir_instr->get_source_class());
            }
            // ret <reg>
            else if (auto ret_instr = ret_ir(instr))
            {
                irstmnt_t return_value = ret_instr->get_return_value();

                if (auto reg = register_ir(return_value))
                    return_value = rename(reg);

                new_instr = std::make_shared<IRRet>(std::dynamic_pointer_cast<IRExpr>(return_value));
            }
            // call <callee>(arg1, arg2, arg3...)
            else if (auto call_instr = call_ir(instr))
            {
                std::vector<irexpr_t> new_args;
                auto args = call_instr->get_args();
                irstmnt_t ret_val = call_instr->get_ret_val();

                for (auto arg : args)
                {
                    new_args.push_back(reg_last_index.at(std::dynamic_pointer_cast<IRReg>(arg)));
                }

                if (auto reg = register_ir(ret_val))
                    ret_val = rename(reg);

                new_instr = std::make_shared<IRCall>(call_instr->get_callee(), call_instr->get_call_type(), new_args);
                std::dynamic_pointer_cast<IRCall>(new_instr)->set_ret_val(std::dynamic_pointer_cast<IRExpr>(ret_val));
            }
            // STORE(A) = B
            else if (auto store_instr = store_ir(instr))
            {
                irstmnt_t destination = store_instr->get_destination();
                irstmnt_t source = store_instr->get_source();
                irstmnt_t index = store_instr->get_index();

                if (auto reg = register_ir(destination))
                    destination = rename(reg);

                if (auto reg = register_ir(source))
                    source = reg_last_index.at(reg);

                if (auto reg = register_ir(index))
                    index = reg_last_index.at(reg);

                new_instr = std::make_shared<IRStore>(std::dynamic_pointer_cast<IRExpr>(destination), std::dynamic_pointer_cast<IRExpr>(source), std::dynamic_pointer_cast<IRExpr>(index), store_instr->get_size());
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