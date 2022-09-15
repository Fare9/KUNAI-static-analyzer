#include "KUNAI/mjolnIR/Analysis/optimizer.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        void Optimizer::add_single_stmnt_pass(one_stmnt_opt_t opt)
        {
            // check if pass is already present
            if (std::find(single_statement_optimization.begin(), single_statement_optimization.end(), opt) != std::end(single_statement_optimization))
                return;
            single_statement_optimization.push_back(opt);
        }

        void Optimizer::add_single_block_pass(one_block_opt_t opt)
        {
            if (std::find(single_block_optimization.begin(), single_block_optimization.end(), opt) != std::end(single_block_optimization))
                return;
            single_block_optimization.push_back(opt);
        }


        void Optimizer::run_analysis(irgraph_t &func)
        {
            auto logger = LOGGER::logger();

            reachingdefinition = std::make_shared<ReachingDefinition>(func);

            auto &blocks = func->get_nodes();

            // run first the instruction optimizer.
            for (size_t i = 0, blk_size = blocks.size(); i < blk_size; i++)
            {
                auto &stmnts = blocks[i]->get_statements();

                for (size_t j = 0, stmnt_size = stmnts.size(); j < stmnt_size; j++)
                {
                    
                    for (auto opt : single_statement_optimization)
                    {
                        irstmnt_t &final_stmnt = stmnts[j];
                        auto result_stmnt = (*opt)(final_stmnt);

                        if (result_stmnt.has_value())
                            stmnts[j] = result_stmnt.value();
                    }
                }
            }

            // second the block optimizers.
            for (size_t i = 0, blk_size = blocks.size(); i < blk_size; i++)
            {
                for (auto opt : single_block_optimization)
                {
                    irblock_t& block = blocks[i];
                    auto result_block = (*opt)(block, func);

                    if (result_block.has_value())
                        blocks[i] = result_block.value();
                } 
            }

            logger->info("Applying Reaching Definition Analysis");

            reachingdefinition->compute();

            // now it's possible to apply analysis that depend on the data flow
        }

        void Optimizer::fallthrough_target_analysis(MJOLNIR::irgraph_t &ir_graph)
        {
            auto nodes = ir_graph->get_nodes();

            for (auto node : nodes)
            {

                if (node->get_number_of_statements() == 0) // security check
                    continue;

                auto last_inst = node->get_statements().back();

                if (MJOLNIR::unconditional_jump_ir(last_inst) != nullptr || MJOLNIR::ret_ir(last_inst) != nullptr)
                    continue;

                for (auto aux : nodes)
                {
                    if (node->get_end_idx() == aux->get_start_idx())
                    {
                        ir_graph->add_uniq_edge(node, aux);
                    }
                }
            }
        }

        void Optimizer::calculate_def_use_and_use_def_analysis(MJOLNIR::irgraph_t &ir_graph,
                                                               reachingdefinition_t &reachingdefinition)
        {
            for (auto &block : ir_graph->get_nodes())
            {
                auto &instructions = block->get_statements();

                for (size_t _size_instr = block->get_number_of_statements(), i = 0; i < _size_instr; i++)
                {
                    auto &instr = instructions.at(i);

                    auto reach_def_instr = reachingdefinition->get_reach_definition_point(block->get_start_idx(), i);

                    // check if there was a reach_def
                    if (!reach_def_instr.has_value())
                        continue;

                    auto reach_def_set = reach_def_instr.value();

                    // check if set is empty
                    if (reach_def_set.empty())
                        continue;

                    // A = B
                    if (auto assign_instr = assign_ir(instr))
                    {
                        assign_instr->invalidate_chains();

                        auto op = assign_instr->get_source();
                        solve_def_use_use_def(op, assign_instr, reach_def_set, ir_graph);
                    }
                    // A = phi(A1, A2, A3, ...)
                    else if (auto phi_instr = phi_ir(instr))
                    {
                        phi_instr->invalidate_chains();

                        auto& params = phi_instr->get_params();
                        for (auto& op : params)
                            solve_def_use_use_def(op.second, phi_instr, reach_def_set, ir_graph);
                    }
                    // A = IRUnaryOp B
                    else if (auto unary_op_instr = unary_op_ir(instr))
                    {
                        unary_op_instr->invalidate_chains();

                        auto op = unary_op_instr->get_op();

                        solve_def_use_use_def(op, unary_op_instr, reach_def_set, ir_graph);
                    }
                    // A = B IRBinaryOp C
                    else if (auto bin_op_instr = bin_op_ir(instr))
                    {
                        bin_op_instr->invalidate_chains();

                        auto op1 = bin_op_instr->get_op1();
                        auto op2 = bin_op_instr->get_op2();

                        solve_def_use_use_def(op1, bin_op_instr, reach_def_set, ir_graph);
                        solve_def_use_use_def(op2, bin_op_instr, reach_def_set, ir_graph);
                    }
                    // CALL (A,B,C,...)
                    else if (auto call_instr = call_ir(instr))
                    {
                        for (auto op : call_instr->get_args())
                        {
                            solve_def_use_use_def(op, call_instr, reach_def_set, ir_graph);
                        }
                    }
                    // A = *B[C]
                    else if (auto load_instr = load_ir(instr))
                    {
                        load_instr->invalidate_chains();

                        auto source = load_instr->get_source();
                        auto index = load_instr->get_index();

                        solve_def_use_use_def(source, load_instr, reach_def_set, ir_graph);

                        if (index)
                            solve_def_use_use_def(index, load_instr, reach_def_set, ir_graph);
                    }
                    // *B[C] = A
                    else if (auto store_instr = store_ir(instr))
                    {
                        store_instr->invalidate_chains();

                        auto source = store_instr->get_source();

                        solve_def_use_use_def(source, store_instr, reach_def_set, ir_graph);
                    }
                    // RET A
                    else if (auto ret_instr = ret_ir(instr))
                    {
                        ret_instr->invalidate_chains();

                        auto ret_value = ret_instr->get_return_value();

                        if (auto reg_value = expr_ir(ret_value))
                            solve_def_use_use_def(reg_value, ret_instr, reach_def_set, ir_graph);
                    }
                    // JCC <condition>
                    else if (auto jcc_instr = conditional_jump_ir(instr))
                    {
                        jcc_instr->invalidate_chains();

                        auto condition = jcc_instr->get_condition();

                        if (auto reg_value = expr_ir(condition))
                            solve_def_use_use_def(reg_value, jcc_instr, reach_def_set, ir_graph);
                    }
                    // BComp A, B
                    else if (auto bcomp_instr = bcomp_ir(instr))
                    {
                        bcomp_instr->invalidate_chains();

                        if (auto reg = bcomp_instr->get_reg1())
                            solve_def_use_use_def(reg, bcomp_instr, reach_def_set, ir_graph);
                        
                        if (auto reg = bcomp_instr->get_reg2())
                            solve_def_use_use_def(reg, bcomp_instr, reach_def_set, ir_graph);
                    }
                    // ZComp A, 0
                    else if (auto zcomp_instr = zcomp_ir(instr))
                    {
                        zcomp_instr->invalidate_chains();

                        if (auto reg = zcomp_instr->get_reg())
                            solve_def_use_use_def(reg, zcomp_instr, reach_def_set, ir_graph);
                    }
                }
            }
        }

        void Optimizer::solve_def_use_use_def(irexpr_t &operand,
                                              irstmnt_t expr,
                                              regdefinitionset_t &reach_def_set,
                                              MJOLNIR::irgraph_t &ir_graph)
        {
            // we need to detect the operand in the reach definition
            // and in case we find it, we will create the def-use and
            // use-def chains.
            for (auto &reach_def_map : reach_def_set)
            {
                // look for the operand in the Reaching definition
                if (reach_def_map.find(operand) != reach_def_map.end())
                {
                    auto &reach_def = reach_def_map.at(operand);

                    // extract where the operand was defined.
                    auto block = std::get<0>(reach_def);
                    auto instr = std::get<1>(reach_def);

                    auto definition_block = ir_graph->get_node_by_start_idx(block);

                    if (!definition_block.has_value())
                        continue;

                    // get the instruction, we will use it to cross-reference both
                    auto definition_instr = definition_block.value()->get_statements().at(instr);

                    // set one use of a definition
                    definition_instr->add_instr_to_use_def_chain(expr);

                    // set one definition of a use
                    expr->add_instr_to_def_use_chain(operand, definition_instr);
                }
            }
        }

        optimizer_t NewDefaultOptimizer()
        {
            auto optimizer = std::make_shared<Optimizer>();

            // single statement optimizers
            optimizer->add_single_stmnt_pass(KUNAI::MJOLNIR::constant_folding);

            // single block optimizers
            optimizer->add_single_block_pass(KUNAI::MJOLNIR::nop_removal);
            optimizer->add_single_block_pass(KUNAI::MJOLNIR::expression_simplifier);
            optimizer->add_single_block_pass(KUNAI::MJOLNIR::instruction_combining);

            return optimizer;
        }
    }
}