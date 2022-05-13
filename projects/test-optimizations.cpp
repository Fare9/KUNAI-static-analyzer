#include <iostream>
#include <memory>

#include "optimizer.hpp"

int
main()
{
    auto optimizer = KUNAI::MJOLNIR::NewDefaultOptimizer();

    auto graph = std::make_shared<KUNAI::MJOLNIR::IRGraph>();

    // Create a couple of blocks
    auto block1 = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

    std::string any_str = "test_passed";

    auto nop = std::make_shared<KUNAI::MJOLNIR::IRNop>();
    auto callee = std::make_shared<KUNAI::MJOLNIR::IRCallee>(0xA, any_str, any_str, 2, any_str, any_str, 0);

    auto const_1 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(2, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "2", 64);
    auto const_2 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(3, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "2", 64);

    auto const_3 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(10, false, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "2", 64);
    auto const_4 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(43, false, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, "2", 64);

    auto temp_reg1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 4);
    auto temp_reg2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x1, any_str, 4);
    auto temp_reg3 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x2, any_str, 4);
    auto temp_reg4 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x4, any_str, 4);

    auto add_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::ADD_OP_T, temp_reg1, const_1, const_2, nullptr, nullptr);
    auto sub_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::SUB_OP_T, temp_reg2, const_1, const_2, nullptr, nullptr);
    auto and_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::AND_OP_T, temp_reg3, const_3, const_4, nullptr, nullptr);
    auto or_instr = std::make_shared<KUNAI::MJOLNIR::IRBinOp>(KUNAI::MJOLNIR::IRBinOp::XOR_OP_T, temp_reg4, const_3, const_4, nullptr, nullptr);

    block1->append_statement_to_block(nop);
    block1->append_statement_to_block(callee);
    block1->append_statement_to_block(add_instr);
    block1->append_statement_to_block(sub_instr);
    block1->append_statement_to_block(and_instr);
    block1->append_statement_to_block(or_instr);

    graph->add_node(block1);

    graph->generate_dot_file("no_simplified");

    optimizer->run_analysis(graph);

    graph->generate_dominator_tree("simplified");

    

    return 0;
}