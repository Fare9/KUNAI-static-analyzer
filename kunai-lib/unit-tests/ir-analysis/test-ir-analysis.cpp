//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer, library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @file test-ir-analysis.cpp
// @brief Test class for the IRManager of MjolnIR

#include "test-ir-analysis.inc"
#include "Kunai/DEX/dex.hpp"
#include "Kunai/Utils/logger.hpp"
#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "MjolnIR/Analysis/MjolnIRManager.hpp"


#include <mlir/IR/AsmState.h>
#include <llvm/Support/CommandLine.h>

#include <memory>

namespace cl = llvm::cl;

static cl::opt<std::string> FilePath("file", llvm::cl::desc("File path"), llvm::cl::value_desc("filename"), llvm::cl::Required);
static cl::opt<std::string> MethodName("method", llvm::cl::desc("Method name"), llvm::cl::value_desc("methodname"), llvm::cl::Required);

int main(int argc, char **argv)
{
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "MjolnIR Dialect\n");

    std::string dex_file_path;
    std::string method_name;

    dex_file_path = FilePath;
    method_name = MethodName;

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

        auto ir_manager = std::make_unique<KUNAI::MjolnIR::MjolnIRManager>(std::move(module_op));

        ir_manager->get_module_op()->dump();

        auto range = ir_manager->get_module_op().getBody();

        mlir::func::FuncOp func;

        for (auto & op : range->getOperations())
        {
            if ((func = mlir::dyn_cast<mlir::func::FuncOp>(op)))
            {
                std::cout << "Function found: " << '\n';
                break;
            }
        }

        if (!func)
            break;
        
        auto & region = func.getBody();

        auto &last_block = region.back();

        auto term = last_block.getTerminator();

        if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(term))
        {
            std::cout << "Slicing operand from instruction:" << std::endl;

            ret->dump();

            auto val = ret.getOperand(0);

            if (!val)
                break;

            std::cout << "Sliced value" << std::endl;

            for (auto entry : ir_manager->sliceValue(val))
            {
                auto BB = entry->getBlock();

                BB->printAsOperand(llvm::errs());

                std::cout << ":" << std::endl;

                entry->dump();
            }
        }
    }
    
}