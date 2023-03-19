//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-xrefs.cpp
// @brief Unit test script for the DEX xrefs creation

#include "test-xrefs.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include <assert.h>

std::vector<std::tuple<std::string, std::string, uint64_t>>
    expected_classes = {
        /// new instance
        {"Ljava/util/Scanner;", "void Main->main(java.lang.String[])", 0},
};

std::vector<std::tuple<std::string, std::string, std::string, uint64_t>>
    expected_values = {
        /// xref_from
        {"Ljava/util/Scanner;->closeV", "LMain;", "main", 268},
        {"Ljava/io/PrintStream;->printlnVL", "LMain;", "main", 136},
        {"Ljava/io/PrintStream;->printlnVL", "LMain;", "main", 150},
};

std::vector<std::tuple<std::string, std::string, std::string, uint64_t>>
    expected_values2 = {
        /// xref_to
        {"void Main->main(java.lang.String[])", "Ljava/util/Scanner;", "<init>", 8},
        {"void Main->main(java.lang.String[])", "Ljava/util/Scanner;", "nextInt", 14},
        {"void Main->main(java.lang.String[])", "Ljava/util/Scanner;", "nextInt", 38}};

int main()
{
    std::string dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-assignment-arith-logic/Main.dex";

    auto logger = KUNAI::LOGGER::logger();

    logger->set_level(spdlog::level::debug);

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    auto analysis = dex->get_analysis(true);

    analysis->create_xrefs();

    auto &classes = analysis->get_classes();

    size_t i = 0;

    for (auto &name_class : classes)
    {
        auto cls = name_class.second.get();

        for (auto &xref : cls->get_xrefnewinstance())
        {
            // logger->debug("New Instance of {} in Method: {} and Offset: {}",
            //     cls->name(),
            //     xref.first->get_full_name(),
            //     xref.second);

            assert((cls->name() == std::get<0>(expected_classes[i])) && "Class name incorrrect for new instance");
            ;
            assert((xref.first->get_full_name() == std::get<1>(expected_classes[i])) && "Method full name incorrect for new instance");
            assert((xref.second == std::get<2>(expected_classes[i])) && "Offset incorrect for new instance");
            i += 1;
        }

        for (auto &xref : cls->get_xrefconstclass())
        {
            // logger->debug("New xref const class of {} in Method: {} and offset: {}",
            //     cls->name(),
            //     xref.first->get_full_name(),
            //     xref.second);
        }
    }

    auto &methods = analysis->get_methods();

    i = 0;

    for (auto &name_method : methods)
    {
        auto method = name_method.second.get();

        for (auto &xref_from : method->get_xreffrom())
        {
            //logger->debug("New xref_from for method {}, from class {} method {} offset {}",
            //              method->get_full_name(),
            //              std::get<0>(xref_from)->name(),
            //              std::get<1>(xref_from)->get_name(),
            //              std::get<2>(xref_from));

            assert((method->get_full_name() == std::get<0>(expected_values[i])) && "Method incorrect new xref_from");
            assert((std::get<0>(xref_from)->name() == std::get<1>(expected_values[i])) && "Class name incorrect new xref_from");
            assert((std::get<1>(xref_from)->get_name() == std::get<2>(expected_values[i])) && "Method name incorrect new xref_from");
            assert((std::get<2>(xref_from) == std::get<3>(expected_values[i])) && "Offset incorrect new xref_from");

            i += 1;

            if (i == 3)
                break;
        }

        if (i == 3)
            break;
    }

    i = 0;

    for (auto &name_method : methods)
    {
        auto method = name_method.second.get();

        for (auto &xref_to : method->get_xrefto())
        {
            // logger->debug("New xref_to from method {}, to class {} method {} offset {}",
            //     method->get_full_name(),
            //     std::get<0>(xref_to)->name(),
            //     std::get<1>(xref_to)->get_name(),
            //     std::get<2>(xref_to));

            assert((method->get_full_name() == std::get<0>(expected_values2[i])) && "Method incorrect new xref_to");
            assert((std::get<0>(xref_to)->name() == std::get<1>(expected_values2[i])) && "Class name incorrect new xref_to");
            assert((std::get<1>(xref_to)->get_name() == std::get<2>(expected_values2[i])) && "Method name incorrect new xref_to");
            assert((std::get<2>(xref_to) == std::get<3>(expected_values2[i])) && "Offset incorrect new xref_to");
        
            i += 1;

            if (i == 3)
                break;
        }

        if (i == 3)
            break;
    }

    return 0;
}