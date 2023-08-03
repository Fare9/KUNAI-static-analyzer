//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRManager.hpp
// @brief

#ifndef ANALYSIS_MJOLNIRMANAGER_HPP
#define ANALYSIS_MJOLNIRMANAGER_HPP

/// MLIR includes
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include "MjolnIR/Dalvik/MjolnIRDialect.hpp"
#include "MjolnIR/Dalvik/MjolnIRTypes.hpp"
#include "MjolnIR/Dalvik/MjolnIROps.hpp"

#include "MjolnIR/Transforms/MjolnIRToOpGraph.hpp"
#include "MjolnIR/Transforms/CfgToScf.hpp"


namespace KUNAI
{
namespace MjolnIR
{
    class MjolnIRManager
    {
        /// @brief pointer to a generated module op to manage
        mlir::OwningOpRef<mlir::ModuleOp> module_op_;

    public:
        MjolnIRManager(mlir::OwningOpRef<mlir::ModuleOp> module_op_) : module_op_(std::move(module_op_))
        {
        }


        mlir::ModuleOp get_module_op()
        {
            return module_op_.get();
        }

        const mlir::ModuleOp get_module_op() const
        {
            return module_op_.get();
        }

        /// @brief Dump the whole module to a file by a given name
        /// @param file_name name where to dump the module
        /// @return true in case of some error, and false if everything was fine.
        bool dump_to_file(std::string file_name);

        /// @brief Generate a CFG from each function from the module of the IR manager.
        /// Dump the output to the provided raw_ostream
        /// @param os raw_ostream where to dump the CFG, by default llvm::errs()
        /// @return result of the pass manager
        mlir::LogicalResult generate_functions_cfg(mlir::raw_ostream &os = llvm::errs());

        /// @brief Decompile the functions from the module but to avoid modifying the
        /// current module op, return a module op with the decompilation.
        /// @return ModuleOp with the decompiled version of the functions.
        std::optional<mlir::ModuleOp> decompile_functions();

        /// @brief Return a vector with all the instructions that set the given value
        /// the function need to go over all the operations in a backward way.
        mlir::SmallVector<mlir::Operation*> sliceValue(mlir::Value value);
    };
}
}


#endif