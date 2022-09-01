#include <iostream>
#include <memory>

#include "KUNAI/mjolnIR/arch/ir_x86.hpp"
#include "KUNAI/mjolnIR/Analysis/optimizer.hpp"
#include "KUNAI/mjolnIR/Analysis/reachingDefinition.hpp"
#include "KUNAI/mjolnIR/Analysis/ir_graph_ssa.hpp"

int
main()
{
    std::cout << "Testing single block optimizations\n";

    auto optimizer = std::make_shared<KUNAI::MJOLNIR::Optimizer>();

    auto graph = KUNAI::MJOLNIR::get_shared_empty_graph();

    // Create a block
    auto block1 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

    std::string any_str = "test_passed";

    auto nop = std::make_shared<KUNAI::MJOLNIR::IRNop>();
    
    // SimplifySubInst
    // X - (X - Y) -> Y
    auto temp_reg1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 4);
    auto temp_reg2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x1, any_str, 4);
    auto reg1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0, KUNAI::MJOLNIR::dalvik_arch, "v0", 4);
    auto reg2 = std::make_shared<KUNAI::MJOLNIR::IRReg>(1, KUNAI::MJOLNIR::dalvik_arch, "v1", 4);
    auto sub_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, temp_reg1, reg1, reg1);
    auto sub_instr2 = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, temp_reg2, temp_reg1, reg2);

    block1->append_statement_to_block(nop);
    block1->append_statement_to_block(sub_instr);
    block1->append_statement_to_block(sub_instr2);

    // SimplifyAddInst
    // (X + Y) - X -> Y
    auto temp_reg3 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x3, any_str, 4);
    auto temp_reg4 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x4, any_str, 4);
    auto reg3 = std::make_shared<KUNAI::MJOLNIR::IRReg>(2, KUNAI::MJOLNIR::dalvik_arch, "v2", 4);
    auto reg4 = std::make_shared<KUNAI::MJOLNIR::IRReg>(3, KUNAI::MJOLNIR::dalvik_arch, "v3", 4);
    auto add_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, temp_reg3, reg3, reg4);
    auto sub_instr3 = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, temp_reg4, temp_reg3, reg3);

    block1->append_statement_to_block(nop);
    block1->append_statement_to_block(add_instr);
    block1->append_statement_to_block(sub_instr3);

    // or (Y - X) + X -> Y
    auto temp_reg5 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x5, any_str, 4);
    auto temp_reg6 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x6, any_str, 4);
    auto reg5 = std::make_shared<KUNAI::MJOLNIR::IRReg>(4, KUNAI::MJOLNIR::dalvik_arch, "v4", 4);
    auto reg6 = std::make_shared<KUNAI::MJOLNIR::IRReg>(5, KUNAI::MJOLNIR::dalvik_arch, "v5", 4);
    auto sub_instr4 = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, temp_reg5, reg6, reg5);
    auto add_instr2 = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, temp_reg6, temp_reg5, reg5);

    block1->append_statement_to_block(nop);
    block1->append_statement_to_block(sub_instr4);
    block1->append_statement_to_block(add_instr2);

    block1->set_start_idx(0);
    block1->set_end_idx(12);

    graph->add_node(block1);

    std::cout << "Generating not simplified graph\n";

    graph->generate_dot_file("no_simplified");

    auto reachingdefinition = std::make_shared<KUNAI::MJOLNIR::ReachingDefinition>(graph);

    reachingdefinition->compute();

    std::cout << "Reaching Definition Analysis:" << std::endl;

    std::cout << *reachingdefinition;

    optimizer->calculate_def_use_and_use_def_analysis(graph, reachingdefinition);

    for (auto& block : graph->get_nodes())
        for (auto& instr : block->get_statements())
            instr->print_use_def_and_def_use_chain();

    // now go for the optimizations
    optimizer->add_single_block_pass(KUNAI::MJOLNIR::nop_removal);
    optimizer->add_single_block_pass(KUNAI::MJOLNIR::expression_simplifier);

    std::cout << "Running analysis and generating simplified graph\n";

    optimizer->run_analysis(graph);

    graph->generate_dot_file("simplified");

    return 0;
}