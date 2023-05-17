//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer, library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-lifter.cpp
// @brief Unit test for the lifting from DEX to MjolnIR

#include "test-lifter.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include "Lifter/MjolnIRLifter.hpp"
#include <memory>

int main(int argc, char ** argv)
{
    std::string dex_file_path;
    std::string method_name;

    if (argc == 1)
    {
        dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-very-simple/classes.dex";
        method_name = "my_add";
    }
    else if (argc == 3)
    {
        dex_file_path = argv[1];
        method_name = argv[2];
    }

    auto logger = KUNAI::LOGGER::logger();

    logger->set_level(spdlog::level::debug);

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;
    
    auto analysis = dex->get_analysis(false);

    const auto &methods = analysis->get_methods();

    for (const auto & method : methods)
    {
        if (method.second->get_name() != method_name)
            continue;

        mlir::DialectRegistry registry;
        
        registry.insert<::mlir::KUNAI::MjolnIR::MjolnIRDialect>();

        mlir::MLIRContext context;
        context.loadAllAvailableDialects();

        KUNAI::MjolnIR::Lifter lifter(context, false);

        auto module_op = lifter.mlirGen(method.second.get());

        module_op->dump();
        
        break;
    }
}