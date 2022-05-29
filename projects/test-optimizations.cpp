#include <iostream>
#include <memory>

#include "optimizer.hpp"
#include "reachingDefinition.hpp"

int
main()
{
    auto optimizer = KUNAI::MJOLNIR::NewDefaultOptimizer();

    auto graph = std::make_shared<KUNAI::MJOLNIR::IRGraph>();

    // Create a block
    auto block1 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

    std::string any_str = "test_passed";

    auto nop = std::make_shared<KUNAI::MJOLNIR::IRNop>();
    auto callee = std::make_shared<KUNAI::MJOLNIR::IRCallee>(0xA, any_str, any_str, 2, any_str, any_str, 0);

    auto const_1 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(2, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "2", 64);
    auto const_2 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(3, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "3", 64);

    auto const_3 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(10, false, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "10", 64);
    auto const_4 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(43, false, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "43", 64);

    auto temp_reg1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 4);
    auto temp_reg2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x1, any_str, 4);
    auto temp_reg3 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x2, any_str, 4);
    auto temp_reg4 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x4, any_str, 4);
    auto temp_reg10 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x10, any_str, 4);

    auto add_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, temp_reg1, const_1, const_2);
    auto sub_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, temp_reg2, const_1, const_2);
    auto and_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::AND_OP_T, temp_reg3, const_3, const_4);
    auto or_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::XOR_OP_T, temp_reg4, const_3, const_4);
    auto assign_instr = std::make_shared<KUNAI::MJOLNIR::IRAssign>(temp_reg10, temp_reg4);

    block1->append_statement_to_block(nop);
    block1->append_statement_to_block(callee);
    block1->append_statement_to_block(add_instr);
    block1->append_statement_to_block(sub_instr);
    block1->append_statement_to_block(and_instr);
    block1->append_statement_to_block(or_instr);
    block1->append_statement_to_block(assign_instr);
    block1->set_start_idx(0);
    block1->set_end_idx(7);

    graph->add_node(block1);

    // create another block

    auto block2 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

    auto temp_reg5 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x5, any_str, 4);
    auto temp_reg6 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x6, any_str, 4);

    auto reg1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0, KUNAI::MJOLNIR::dalvik_arch, "v0", 4);
    auto reg2 = std::make_shared<KUNAI::MJOLNIR::IRReg>(1, KUNAI::MJOLNIR::dalvik_arch, "v1", 4);
    
    auto const_5 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(0, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "0", 32);
    auto const_6 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(1, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "1", 32);

    auto add_reg_zero = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, temp_reg5, reg1, const_5);
    auto add_zero_reg = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, temp_reg5,const_5, reg1);

    auto multiply_reg_one = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::S_MUL_OP_T, temp_reg6, reg2, const_6);
    auto multiply_one_reg = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::S_MUL_OP_T, temp_reg6, const_6, reg2);

    block2->append_statement_to_block(add_reg_zero);
    block2->append_statement_to_block(add_zero_reg);
    block2->append_statement_to_block(multiply_reg_one);
    block2->append_statement_to_block(multiply_one_reg);
    block2->set_start_idx(8);
    block2->set_end_idx(12);

    graph->add_node(block2);

    graph->add_edge(block1, block2);

    graph->generate_dot_file("no_simplified");

    auto reachingdefinition = std::make_shared<KUNAI::MJOLNIR::ReachingDefinition>(graph);

    reachingdefinition->compute();

    std::cout << "Reaching Definition Analysis:" << std::endl;

    std::cout << *reachingdefinition;

    optimizer->calculate_def_use_and_use_def_analysis(graph, reachingdefinition);

    for (auto& block : graph->get_nodes())
        for (auto& instr : block->get_statements())
            instr->print_use_def_and_def_use_chain();

    optimizer->run_analysis(graph);

    graph->generate_dot_file("simplified");

    const auto& last_block = graph->get_nodes().back();

    auto last_reach_definition = reachingdefinition->get_reach_definition_point(last_block->get_start_idx(), last_block->get_number_of_statements());

    return 0;
}