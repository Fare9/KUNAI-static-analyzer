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

            dominance_tree = compute_immediate_dominators();

            collect_var_assign();
            insert_phi_node();

            auto &first_node = get_nodes()[0];

            search(first_node);
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

            for (const auto &p : var_block_map)
            {
                const irreg_t &reg = p.first;

                for (auto &block : p.second)
                    work_list.push_back(block);

                while (!work_list.empty())
                {
                    auto &block = work_list.front();
                    work_list.erase(work_list.begin());

                    for (auto &df_block : dominance_frontier[block])
                    {
                        if (inserted[df_block] != reg)
                        {
                            // add phi node
                            inserted[df_block] = reg;

                            auto phi_instr = std::make_shared<IRPhi>();
                            phi_instr->add_result(reg);

                            df_block->add_statement_at_beginning(phi_instr);

                            df_block->set_phi_node();

                            // finally add the block from dominance_frontier
                            // into the worklist
                            work_list.push_back(df_block);
                        }
                    }
                }
            }
        }

        void IRGraphSSA::search(const irblock_t &v)
        {
            std::list<irreg_t> p;

            auto &statements = v->get_statements();

            // process each statement of the block
            for (size_t v_size = statements.size(), i = 0; i < v_size; i++)
            {
                auto &instr = statements[i];

                auto new_instr = translate_instruction(instr, p);

                if (new_instr != instr)
                    statements[i] = new_instr;
            }
            // process the phi statements from the successors
            auto &succs = get_successors(v);
            for (auto &w : succs)
            {
                // if the next block does not contain
                // a phi node, just continue, avoid
                // all the other calculations
                if (!w->contains_phi_node())
                    continue;

                // extract which_pred is v for w
                // this will take by index which
                // predecessor is v from w
                auto &preds = get_predecessors(w);
                auto it = find(preds.begin(), preds.end(), v);

                int j = -1;

                if (it != preds.end())
                    j = it - preds.begin();

                // now look for phi functions.
                auto &w_stmnts = w->get_statements();
                for (auto &w_stmnt : w_stmnts)
                {
                    if (auto phi_instr = phi_ir(w_stmnt))
                    {
                        // trick to fill the parameters from the PHI function
                        // extract the result register, and turn it to a non SSA form
                        // if needed, then assign the register to the phi statement
                        // as one of the parameters.

                        irreg_t reg = std::dynamic_pointer_cast<IRReg>(phi_instr->get_result());

                        if (reg->get_sub_id() != -1)
                            reg = ssa_to_non_ssa_form[reg];

                        phi_instr->get_params()[j] = S[reg].top();
                    }
                }
            }

            // go through each child from the dominance tree
            for (auto &doms : dominance_tree)
                // check that current block strictly
                // dominates the next one to analyze
                if (doms.second == v)
                {
                    auto &child = doms.first;
                    search(child);
                }

            // now POP all the defined variables here!
            for (auto x : p)
                S[x].pop();
        }

        irstmnt_t IRGraphSSA::translate_instruction(irstmnt_t &instr, std::list<irreg_t> &p)
        {
            irstmnt_t new_instr = instr;

            // A = B
            if (auto assign_instr = assign_ir(instr))
            {
                irstmnt_t destination = assign_instr->get_destination();
                irstmnt_t source = assign_instr->get_source();

                if (auto reg = register_ir(source))
                    source = S[reg].top();

                if (auto reg = register_ir(destination))
                    destination = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRAssign>(std::dynamic_pointer_cast<IRExpr>(destination), std::dynamic_pointer_cast<IRExpr>(source));
            }
            // A = IRUnaryOp B
            else if (auto unary_instr = unary_op_ir(instr))
            {
                irstmnt_t result = unary_instr->get_result();
                irstmnt_t op = unary_instr->get_op();

                if (auto reg = register_ir(op))
                    op = S[reg].top();

                if (auto reg = register_ir(result))
                    result = create_new_ssa_reg(reg, p);

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

                if (auto reg = register_ir(op1))
                    op1 = S[reg].top();

                if (auto reg = register_ir(op2))
                    op2 = S[reg].top();

                if (auto reg = register_ir(result))
                    result = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRBinOp>(binary_instr->get_bin_op_type(), std::dynamic_pointer_cast<IRExpr>(result), std::dynamic_pointer_cast<IRExpr>(op1), std::dynamic_pointer_cast<IRExpr>(op2));
            }
            // A = LOAD(B[INDEX])
            else if (auto load_instr = load_ir(instr))
            {
                irstmnt_t destination = load_instr->get_destination();
                irstmnt_t source = load_instr->get_source();
                irstmnt_t index = load_instr->get_index();

                if (auto reg = register_ir(source))
                    source = S[reg].top();

                if (auto reg = register_ir(index))
                    index = S[reg].top();

                if (auto reg = register_ir(destination))
                    destination = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRLoad>(std::dynamic_pointer_cast<IRExpr>(destination), std::dynamic_pointer_cast<IRExpr>(source), std::dynamic_pointer_cast<IRExpr>(index), load_instr->get_size());
            }
            // A = New Class
            else if (auto new_ir_instr = new_ir(instr))
            {
                irstmnt_t result = new_ir_instr->get_result();

                if (auto reg = register_ir(result))
                    result = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRNew>(std::dynamic_pointer_cast<IRExpr>(result), new_ir_instr->get_source_class());
            }
            // ret <reg>
            else if (auto ret_instr = ret_ir(instr))
            {
                irstmnt_t return_value = ret_instr->get_return_value();

                if (auto reg = register_ir(return_value))
                    return_value = create_new_ssa_reg(reg, p);

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
                    auto reg = std::dynamic_pointer_cast<IRReg>(arg);
                    new_args.push_back(S[reg].top());
                }

                if (auto reg = register_ir(ret_val))
                    ret_val = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRCall>(call_instr->get_callee(), call_instr->get_call_type(), new_args);
                std::dynamic_pointer_cast<IRCall>(new_instr)->set_ret_val(std::dynamic_pointer_cast<IRExpr>(ret_val));
            }
            // STORE(A) = B
            else if (auto store_instr = store_ir(instr))
            {
                irstmnt_t destination = store_instr->get_destination();
                irstmnt_t source = store_instr->get_source();
                irstmnt_t index = store_instr->get_index();

                if (auto reg = register_ir(source))
                    source = S[reg].top();

                if (auto reg = register_ir(index))
                    index = S[reg].top();

                if (auto reg = register_ir(destination))
                    destination = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRStore>(std::dynamic_pointer_cast<IRExpr>(destination), std::dynamic_pointer_cast<IRExpr>(source), std::dynamic_pointer_cast<IRExpr>(index), store_instr->get_size());
            }
            // ZComp
            else if (auto zcomp = zcomp_ir(instr))
            {
                irstmnt_t op = zcomp->get_reg();

                if (auto reg = register_ir(op))
                    op = S[reg].top();

                new_instr = std::make_shared<IRZComp>(zcomp->get_comparison(), zcomp->get_result(), std::dynamic_pointer_cast<IRExpr>(op));
            }
            // BComp
            else if (auto bcomp = bcomp_ir(instr))
            {
                irstmnt_t op1 = bcomp->get_reg1();
                irstmnt_t op2 = bcomp->get_reg2();

                if (auto reg = register_ir(op1))
                    op1 = S[reg].top();

                if (auto reg = register_ir(op2))
                    op2 = S[reg].top();

                new_instr = std::make_shared<IRBComp>(bcomp->get_comparison(), bcomp->get_result(), std::dynamic_pointer_cast<IRExpr>(op1), std::dynamic_pointer_cast<IRExpr>(op2));
            }
            // Phi node (only the result)
            if (auto phi_instr = phi_ir(instr))
            {
                irstmnt_t result = phi_instr->get_result();

                if (auto reg = register_ir(result))
                    result = create_new_ssa_reg(reg, p);

                new_instr = std::make_shared<IRPhi>();

                auto aux = std::dynamic_pointer_cast<IRPhi>(new_instr);
                aux->add_result(std::dynamic_pointer_cast<IRExpr>(result));

                for (auto param : phi_instr->get_params())
                {
                    aux->add_param(param.second, param.first);
                }
            }

            return new_instr;
        }

        irreg_t IRGraphSSA::create_new_ssa_reg(irreg_t old_reg, std::list<irreg_t> &p)
        {
            irreg_t new_reg;

            if (C.find(old_reg) == C.end())
                C[old_reg] = 0;

            new_reg = std::make_shared<IRReg>(old_reg->get_id(),
                                              C[old_reg],
                                              old_reg->get_current_arch(),
                                              old_reg->to_string() + "." + std::to_string(C[old_reg]),
                                              old_reg->get_type_size());
            // save last index of the register
            C[old_reg]++;
            // save all the references to new registers
            // from old one
            S[old_reg].push(new_reg);
            // save the old register from the newer one
            ssa_to_non_ssa_form[new_reg] = old_reg;

            p.push_back(old_reg);

            return new_reg;
        }

    }
}