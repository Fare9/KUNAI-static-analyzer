#include <iostream>
#include <memory>
#include <string>
#include <cassert>

#include "ir_grammar.hpp"
#include "utils.hpp"

int
main()
{
    auto logger = KUNAI::LOGGER::logger();
    // Set the logging level in spdlog, we set to debug
    spdlog::set_level(spdlog::level::debug);

    int number_of_test_passed = 0;

    std::string any_str = "test_passed";
    auto callee = std::make_shared<KUNAI::MJOLNIR::IRCallee>(0xA, any_str, any_str, 2, any_str, any_str, 0);
    auto callee2 = std::make_shared<KUNAI::MJOLNIR::IRCallee>(0xA, any_str, any_str, 2, any_str, any_str, 0);

    logger->debug("Checking two callee objects");
    assert(callee->equal(callee2));
    number_of_test_passed++;
    logger->debug(callee->to_string());


    auto class1 = std::make_shared<KUNAI::MJOLNIR::IRClass>(any_str, any_str, 0);
    auto class2 = std::make_shared<KUNAI::MJOLNIR::IRClass>(any_str, any_str, 0);

    logger->debug("Checking two classes");
    assert(class1->equal(class2));
    number_of_test_passed++;
    logger->debug(class1->to_string());


    auto str1 = std::make_shared<KUNAI::MJOLNIR::IRString>(any_str, any_str, 0);
    auto str2 = std::make_shared<KUNAI::MJOLNIR::IRString>(any_str, any_str, 0);
    
    logger->debug("Checking two strings");
    assert(str1->equal(str2));
    number_of_test_passed++;
    logger->debug(str1->to_string());

    auto mem1 = std::make_shared<KUNAI::MJOLNIR::IRMemory>(0x0, 0x0, KUNAI::MJOLNIR::IRMemory::LE_ACCESS, any_str, 0);
    auto mem2 = std::make_shared<KUNAI::MJOLNIR::IRMemory>(0x0, 0x0, KUNAI::MJOLNIR::IRMemory::LE_ACCESS, any_str, 0);
    
    logger->debug("Checking two memory addresses");
    assert(mem2->equal(mem1));
    number_of_test_passed++;
    logger->debug(mem1->to_string());

    auto const_int1 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(0x0, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, any_str, 0);
    auto const_int2 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(0x0, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, any_str, 0);

    logger->debug("Checking two const ints");
    assert(const_int1->equal(const_int2));
    number_of_test_passed++;
    logger->debug(const_int1->to_string());


    auto temp_reg1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 0);
    auto temp_reg2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 0);

    logger->debug("Checking temporary registers");
    assert(temp_reg1->equal(temp_reg2));
    number_of_test_passed++;
    logger->debug(temp_reg1->to_string());

    auto reg1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0x0, KUNAI::MJOLNIR::x86_arch, any_str, 0);
    auto reg2 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0x0, KUNAI::MJOLNIR::x86_arch, any_str, 0);

    logger->debug("Checking registers");
    assert(reg1->equal(reg2));
    number_of_test_passed++;
    logger->debug(reg1->to_string());

    std::shared_ptr<KUNAI::MJOLNIR::IRType> type1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0x0, KUNAI::MJOLNIR::dalvik_arch, any_str, 0);
    std::shared_ptr<KUNAI::MJOLNIR::IRType> type2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 0);

    logger->debug("Checking two types are different");
    assert(!type1->equal(type2));
    number_of_test_passed++;
    logger->debug(type1->to_string());
    logger->debug(type2->to_string());

    return (number_of_test_passed == 8) ? 0 : 1;
}