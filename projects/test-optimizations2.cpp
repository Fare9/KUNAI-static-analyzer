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


    // Y = X + <INT>
    // Z = Y + <INT>
    // =============
    // Z = X + (<INT> + <INT>)
    auto reg7 = std::make_shared<KUNAI::MJOLNIR::IRReg>(7, KUNAI::MJOLNIR::dalvik_arch, "v7", 4);
    auto reg8 = std::make_shared<KUNAI::MJOLNIR::IRReg>(8, KUNAI::MJOLNIR::dalvik_arch, "v8", 4);
    auto reg9 = std::make_shared<KUNAI::MJOLNIR::IRReg>(9, KUNAI::MJOLNIR::dalvik_arch, "v9", 4);
    auto const1 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(25, false, KUNAI::MJOLNIR::IRType::LE_ACCESS, "25", 4);
    auto const2 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(100, false, KUNAI::MJOLNIR::IRType::LE_ACCESS, "100", 4);

    auto add_instr3 = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, reg8, reg7, const1);
    auto add_instr4 = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, reg9, reg8, const2);

    block1->append_statement_to_block(add_instr3);
    block1->append_statement_to_block(add_instr4);


    // (A | (B ^ C)) ^ ((A ^ C) ^ B)
    auto A = std::make_shared<KUNAI::MJOLNIR::IRReg>(20, KUNAI::MJOLNIR::dalvik_arch, "v20", 4);
    auto B = std::make_shared<KUNAI::MJOLNIR::IRReg>(21, KUNAI::MJOLNIR::dalvik_arch, "v21", 4);
    auto C = std::make_shared<KUNAI::MJOLNIR::IRReg>(22, KUNAI::MJOLNIR::dalvik_arch, "v22", 4);

    auto t_xor1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(30, "t30", 4);
    auto t_or = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(31, "t31", 4);
    auto t_xor2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(32, "t32", 4);
    auto t_xor3 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(33, "t33", 4);
    auto res = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(34, "t34", 4);

    auto first_xor = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::XOR_OP_T, t_xor1, B, C);
    auto first_or = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::OR_OP_T, t_or, A, t_xor1);
    auto second_xor = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::XOR_OP_T, t_xor2, A, C);
    auto third_xor = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::XOR_OP_T, t_xor3, t_xor2, B);
    auto last_xor = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::XOR_OP_T, res, t_or, t_xor3);

    graph->set_last_temporal(34);

    block1->append_statement_to_block(first_xor);
    block1->append_statement_to_block(first_or);
    block1->append_statement_to_block(second_xor);
    block1->append_statement_to_block(third_xor);
    block1->append_statement_to_block(last_xor);


    block1->set_start_idx(0);
    block1->set_end_idx(14);

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
    optimizer->add_single_block_pass(KUNAI::MJOLNIR::instruction_combining);

    std::cout << "Running analysis and generating simplified graph\n";

    optimizer->run_analysis(graph);

    graph->generate_dot_file("simplified");

    return 0;
}