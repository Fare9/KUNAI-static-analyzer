//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRManager.cpp

#include "MjolnIR/Analysis/MjolnIRManager.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <stack>

using namespace KUNAI::MjolnIR;

bool MjolnIRManager::dump_to_file(std::string file_name)
{
    std::error_code error;
    mlir::OpPrintingFlags printFlags;
    llvm::raw_fd_ostream fileStream(file_name, error);

    if (error)
        return true;

    module_op_->print(fileStream, printFlags);
    return false;
}

mlir::LogicalResult MjolnIRManager::generate_functions_cfg(mlir::raw_ostream &os)
{
    mlir::PassManager pm(module_op_.get()->getName());

    pm.addNestedPass<mlir::func::FuncOp>(KUNAI::MjolnIR::createMjolnIROpGraphPass(os));

    return pm.run(*module_op_);
}

std::optional<mlir::ModuleOp> MjolnIRManager::decompile_functions()
{
    auto module_op_copy_ = module_op_->clone();

    mlir::PassManager pm(module_op_copy_->getName());

    pm.addNestedPass<mlir::func::FuncOp>(KUNAI::MjolnIR::createMjolnIRCfgToScfgPass());

    if (mlir::failed(pm.run(module_op_copy_)))
        return std::nullopt;

    return module_op_copy_;
}

mlir::SmallVector<mlir::Operation *> MjolnIRManager::sliceValue(mlir::Value value)
{
    mlir::SmallVector<mlir::Operation *, 4> operations;
    mlir::SetVector<mlir::Operation *> visited;
    std::stack<mlir::Operation *> worklist;

    auto add_argument_block_definition = [&](mlir::Value value)
    {
        auto parent_block = value.getParentBlock();

        assert(parent_block && "Parent block does not exist");

        auto arguments = parent_block->getArguments();

        auto it = std::ranges::find(arguments, value);

        ///if (it == arguments.end())
        ///    return;
        assert((it != arguments.end()) && "Argument not found");

        auto index = it - arguments.begin();

        for (auto pred : parent_block->getPredecessors())
        {
            auto term = pred->getTerminator();

            if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(term))
            {
                auto param = br.getOperand(index);

                worklist.push(param.getDefiningOp());
            }
            else if (auto cond_br = mlir::dyn_cast<mlir::cf::CondBranchOp>(term))
            {
                if (cond_br.getTrueDest() == parent_block)
                {
                    auto param = cond_br.getTrueOperand(index);
                    worklist.push(param.getDefiningOp());
                }
                else if (cond_br.getFalseDest() == parent_block)
                {
                    auto param = cond_br.getFalseOperand(index);
                    worklist.push(param.getDefiningOp());
                }
                else
                    assert(true && "Conditional branch is incorrect...");
            }
        }
    };

    /// the first operation from the work list will be
    /// the one where the value is defnied
    auto def_op = value.getDefiningOp();

    if (def_op)
        worklist.push(def_op);
    else
        add_argument_block_definition(value);

    while (!worklist.empty())
    {
        auto op = worklist.top();
        worklist.pop();

        if (visited.contains(op))
            continue;

        // set it as visited
        visited.insert(op);

        for (auto operand : op->getOperands())
        {
            def_op = operand.getDefiningOp();
            if (def_op)
                worklist.push(def_op);
            else
                add_argument_block_definition(operand);
        }

        operations.emplace_back(op);
    }

    return operations;
}