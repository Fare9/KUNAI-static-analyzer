//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer, library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-lifter.cpp
// @brief Unit test for the lifting from DEX to MjolnIR

#include "test-lifter.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "MjolnIR/Transforms/MjolnIRToOpGraph.hpp"
#include "MjolnIR/Transforms/CfgToScf.hpp"
#include <memory>

#include <mlir/Support/FileUtilities.h>

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/ViewOpGraph.h>

#include <llvm/Support/CommandLine.h>

namespace cl = llvm::cl;



int main(int argc, char **argv)
{
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "MjolnIR Dialect\n");

    std::string dex_file_path;
    std::string method_name;

    //if (argc == 1)
    //{
    dex_file_path = std::string(KUNAI_TEST_FOLDER) + "/test-loop/classes.dex";
    method_name = "test";
    //}
    /*
    else if (argc == 3)
    {
        dex_file_path = argv[1];
        method_name = argv[2];
    }
    */
    auto logger = KUNAI::LOGGER::logger();

    logger->set_level(spdlog::level::debug);

    auto dex = KUNAI::DEX::Dex::parse_dex_file(dex_file_path);

    if (!dex->get_parsing_correct())
        return -1;

    auto analysis = dex->get_analysis(false);

    const auto &methods = analysis->get_methods();

    for (const auto &method : methods)
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

        std::string filePath = "./output.mlir";
        std::error_code error;
        llvm::raw_fd_ostream fileStream(filePath, error);

        if (error)
        {
            llvm::errs() << "Error opening file: " << error.message() << "\n";
            // Handle the error appropriately
        }
        else
        {
            mlir::OpPrintingFlags printFlags;
            module_op->print(fileStream, printFlags);
            llvm::outs() << "MLIR module written to file: " << filePath << "\n";

            mlir::PassManager pm(module_op.get()->getName());

            //pm.addPass(mlir::createPrintOpGraphPass());
            //pm.addNestedPass<mlir::func::FuncOp>(KUNAI::MjolnIR::createMjolnIROpGraphPass());
            pm.addNestedPass<mlir::func::FuncOp>(KUNAI::MjolnIR::createMjolnIRCfgToScfgPass());
            // Apply any generic pass manager command line options and run the pipeline.
            if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
                return 4;

            if (mlir::failed(pm.run(*module_op)))
                return 5;

            module_op->dump();            
        }
    }
}