#include <iostream>
#include <memory>
#include <string>

#include "ir_grammar.hpp"


int
main()
{
    int number_of_test_passed = 0;

    std::string any_str = "dorime";
    auto callee = std::make_shared<KUNAI::MJOLNIR::IRCallee>(0xA, any_str, any_str, 2, any_str, any_str, 0);
    auto callee2 = std::make_shared<KUNAI::MJOLNIR::IRCallee>(0xA, any_str, any_str, 2, any_str, any_str, 0);

    if (callee->equal(callee2))
    {
        number_of_test_passed++;
        std::cout << callee->to_string() << std::endl;
    }

    auto class1 = std::make_shared<KUNAI::MJOLNIR::IRClass>(any_str, any_str, 0);
    auto class2 = std::make_shared<KUNAI::MJOLNIR::IRClass>(any_str, any_str, 0);

    if (class1->equal(class2))
    {
        number_of_test_passed++;
        std::cout << class1->to_string() << std::endl;
    }

    auto str1 = std::make_shared<KUNAI::MJOLNIR::IRString>(any_str, any_str, 0);
    auto str2 = std::make_shared<KUNAI::MJOLNIR::IRString>(any_str, any_str, 0);
    
    if (str1->equal(str2))
    {
        number_of_test_passed++;
        std::cout << str1->to_string() << std::endl;
    }

    auto mem1 = std::make_shared<KUNAI::MJOLNIR::IRMemory>(0x0, 0x0, KUNAI::MJOLNIR::IRMemory::LE_ACCESS, any_str, 0);
    auto mem2 = std::make_shared<KUNAI::MJOLNIR::IRMemory>(0x0, 0x0, KUNAI::MJOLNIR::IRMemory::LE_ACCESS, any_str, 0);
    
    if (mem2->equal(mem1))
    {
        number_of_test_passed++;
        std::cout << mem1->to_string() << std::endl;
    }

    auto const_int1 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(0x0, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, any_str, 0);
    auto const_int2 = std::make_shared<KUNAI::MJOLNIR::IRConstInt>(0x0, true, KUNAI::MJOLNIR::IRConstInt::LE_ACCESS, any_str, 0);

    if (const_int1->equal(const_int2))
    {
        number_of_test_passed++;
        std::cout << const_int1->to_string() << std::endl;
    }

    auto temp_reg1 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 0);
    auto temp_reg2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 0);

    if (temp_reg1->equal(temp_reg2))
    {
        number_of_test_passed++;
        std::cout << temp_reg1->to_string() << std::endl;
    }

    auto reg1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0x0, KUNAI::MJOLNIR::x86_arch, any_str, 0);
    auto reg2 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0x0, KUNAI::MJOLNIR::x86_arch, any_str, 0);

    if (reg1->equal(reg2))
    {
        number_of_test_passed++;
        std::cout << reg1->to_string() << std::endl;
    }

    std::shared_ptr<KUNAI::MJOLNIR::IRType> type1 = std::make_shared<KUNAI::MJOLNIR::IRReg>(0x0, KUNAI::MJOLNIR::dalvik_arch, any_str, 0);
    std::shared_ptr<KUNAI::MJOLNIR::IRType> type2 = std::make_shared<KUNAI::MJOLNIR::IRTempReg>(0x0, any_str, 0);

    if (!type1->equal(type2))
    {
        number_of_test_passed++;
        std::cout << "Dorime" << std::endl;
    }

    return number_of_test_passed;
}