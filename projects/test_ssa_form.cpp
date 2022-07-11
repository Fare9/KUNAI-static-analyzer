#include <iostream>
#include <memory>

#include "KUNAI/mjolnIR/ir_grammar.hpp"
#include "KUNAI/mjolnIR/ir_graph.hpp"
#include "KUNAI/mjolnIR/Analysis/ir_graph_ssa.hpp"

int
main()
{
    auto graph = KUNAI::MJOLNIR::get_shared_empty_graph();

    auto reg1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(KUNAI::MJOLNIR::rax, KUNAI::MJOLNIR::x86_arch, "", 0);
    auto reg2 = std::make_shared<KUNAI::MJOLNIR::IRReg>(KUNAI::MJOLNIR::rbx, KUNAI::MJOLNIR::x86_arch, "", 0);
    auto temp_reg1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, "t0", 4);


    auto block1 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();
    auto block2 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();
    auto block3 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();
    auto block4 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

    auto const_1 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(2, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "2", 64);
    auto const_2 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(3, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "3", 64);
    auto const_3 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(4, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "4", 64);
    auto const_4 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(5, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "5", 64);


    // create block1
    auto assign_instr = std::make_shared<KUNAI::MJOLNIR::IRAssign>(reg1, const_1);
    auto add_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, reg2, reg1, const_2);
    auto comparison = std::make_shared<KUNAI::MJOLNIR::IRZComp>(KUNAI::MJOLNIR::IRZComp::EQUAL_ZERO_T, temp_reg1, reg1);
    auto jcc_cond = std::make_shared<KUNAI::MJOLNIR::IRCJmp>(0x6, temp_reg1, block3, block2);

    block1->append_statement_to_block(assign_instr);
    block1->append_statement_to_block(add_instr);
    block1->append_statement_to_block(comparison);
    block1->append_statement_to_block(jcc_cond);
    block1->set_start_idx(0);
    block1->set_end_idx(3);

    // create block2 (fallthrough)
    auto sub_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, reg1, reg1, const_3);
    auto jmp_instr = std::make_shared<KUNAI::MJOLNIR::IRUJmp>(0x7, block4);
    block2->append_statement_to_block(sub_instr);
    block2->append_statement_to_block(jmp_instr);
    block2->set_start_idx(4);
    block2->set_end_idx(5);

    // create block3 (target)
    auto inc_instr = std::make_shared<KUNAI::MJOLNIR::IRUnaryOp>(KUNAI::MJOLNIR::IRUnaryOp::INC_OP_T, reg1,reg1);
    block3->append_statement_to_block(inc_instr);
    block3->set_start_idx(6);
    block3->set_end_idx(6);
    
    // create block4 (join node)
    auto last_assign = std::make_shared<KUNAI::MJOLNIR::IRAssign>(reg2, reg1);
    block4->append_statement_to_block(last_assign);
    block4->set_start_idx(7);
    block4->set_end_idx(7);

    // create the graph
    graph->add_node(block1);
    graph->add_node(block2);
    graph->add_node(block3);
    graph->add_node(block4);
    
    graph->add_edge(block1, block2);
    graph->add_edge(block1, block3);
    graph->add_edge(block2, block4);
    graph->add_edge(block3, block4);

    // print the graph
    graph->generate_dot_file("no_ssa_graph");

    graph->generate_dominator_tree("no_ssa_graph_dominators");

    auto graph_ssa = std::make_shared<KUNAI::MJOLNIR::IRGraphSSA>(graph);

    graph_ssa->generate_dot_file("ssa_graph");
}