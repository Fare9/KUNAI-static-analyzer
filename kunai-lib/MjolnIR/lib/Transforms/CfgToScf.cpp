//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file CfgToScf.cpp

#include "MjolnIR/Transforms/CfgToScf.hpp"
#include "MjolnIR/Dalvik/MjolnIROps.hpp"

#include <cassert>

#include <mlir/Dialect/Arith/IR/Arith.h>
/// for checking control flow operations
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
/// for using operations from Func dialect
#include <mlir/Dialect/Func/IR/FuncOps.h>
/// for creating operations from scf dialect
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/RegionUtils.h>

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace
{
    static void eraseBlocks(mlir::PatternRewriter &rewriter,
                            llvm::ArrayRef<mlir::Block *> blocks)
    {
        for (auto block : blocks)
        {
            assert(nullptr != block);
            block->dropAllDefinedValueUses();
        }
        for (auto block : blocks)
            rewriter.eraseBlock(block);
    }

    /// @brief Check that all the given blocks are different
    /// @param blocks arrayre fof blocks
    /// @return boolean indicating if all blocks are different
    static bool areBlocksDifferent(llvm::ArrayRef<mlir::Block *> blocks)
    {
        /// go over the blocks
        for (auto &&[i, block1] : llvm::enumerate(blocks))
        {
            assert(nullptr != block1);
            /// get next blocks
            for (auto block2 : blocks.drop_front(i + 1))
            {
                assert(nullptr != block2);
                if (block1 == block2)
                    return false;
            }
        }
        return true;
    }

    /// The next RewritePattern from MLIR will obtain CFGs like
    /// the next two figures:
    ///       BB1                              BB1
    ///      /   \                              | \
    ///     /     \                             |  \
    ///   BB2     BB3                           |  BB2
    ///     \     /                             |   /
    ///      \   /                              |  /
    ///       BB4                               BB3
    ///     Figure 1                          Figure 2
    /// And from them it generates a `scf.if` operation that will
    /// have the shape:
    ///        if condition then;
    ///            Region of Code...
    ///        [else...]

    /// @brief This class will act as rewrite pattern for the if block
    struct ScfIfRewriteOneExit : mlir::OpRewritePattern<mlir::cf::CondBranchOp>
    {
        using OpRewritePattern::OpRewritePattern;

        mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op,
                                            mlir::PatternRewriter &rewriter) const override
        {
            if (!op.getTrueDest() || // no true destination?
                !op.getFalseDest()   // no false destination?
            )
                return mlir::failure();

            /// Two utilities that will be useful to obtain the destination
            /// and the operands from the destination

            /// @brief Get the destination block from operation
            /// @param trueDest get the true or the false destination
            auto getDest = [&](bool trueDest) -> mlir::Block *
            {
                return trueDest ? op.getTrueDest() : op.getFalseDest();
            };

            /// @brief Get the operands from the destination blocks of the operation
            /// @param trueDest get the true or false branch operands
            auto getOperands = [&](bool trueDest) -> mlir::Operation::operand_range
            {
                return trueDest ? op.getTrueOperands() : op.getFalseOperands();
            };

            /// get location for future use
            auto loc = op.getLoc();

            /// create a fake return block for checking and returning values
            auto returnBlock = reinterpret_cast<mlir::Block *>(1);

            /// We will go over the true block and then the false block
            /// and in case there's an error we will go first the false
            /// block and then the true block
            for (auto reverse : {false, true})
            {
                /// get first the true block
                auto trueBlock = getDest(!reverse);

                /// @brief lambda for getting the next block of
                /// and retrieve the next block
                auto getNextBlock = [&](mlir::Block *block) -> mlir::Block *
                {
                    assert(nullptr != block);

                    auto term = block->getTerminator();

                    if (auto br = mlir::dyn_cast_or_null<mlir::cf::BranchOp>(term))
                        return br.getDest();

                    if (auto ret = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(term))
                        return returnBlock;

                    return nullptr;
                };

                /// the next block of a true block, must be the
                /// joiningBlock
                auto joiningBlock = getNextBlock(trueBlock);
                if (nullptr == joiningBlock)
                    continue; // if null, it's weird, try in the other way around...

                /// get the false block
                auto falseBlock = getDest(reverse);

                /// check that falseBlock exist, for doing that we do two checks
                ///     - falseBlock is the joining node
                ///     - next(falseBlock) is the joining node
                if (falseBlock != joiningBlock &&
                    getNextBlock(falseBlock) != joiningBlock)
                    continue;

                /// get the starting node from the conditional block
                auto startBlock = op.getOperation()->getBlock();

                /// now check the three blocks are different, in case
                /// they are the same, maybe we are in a loop...
                if (!areBlocksDifferent({startBlock, trueBlock, joiningBlock}))
                    continue;

                /// are trueBlock and falseBlock equals? Maybe startBlock
                /// points to same block
                if (trueBlock == falseBlock)
                    continue;

                mlir::Value cond = op.getCondition();
                /// in case we are going first over the false block
                /// and then the true block
                if (reverse)
                {
                    /// get a mlir i1 type
                    auto i1 = mlir::IntegerType::get(op.getContext(), 1);
                    /// generate a value 1
                    auto one = rewriter.create<mlir::arith::ConstantOp>(
                        loc, mlir::IntegerAttr::get(i1, 1));
                    /// now inverse the condition using an XOR operation
                    cond = rewriter.create<mlir::arith::XOrIOp>(loc, cond, one);
                }

                /// Mapper for the operations
                mlir::IRMapping mapper;
                /// vector to store yield values
                llvm::SmallVector<mlir::Value> yieldVals;

                /// Copy a block as a yield operation
                auto copyBlock = [&](mlir::OpBuilder &builder, mlir::Location loc,
                                     mlir::Block &block, mlir::ValueRange args)
                {
                    /// check given args and number of arguments are correct
                    assert(args.size() == block.getNumArguments());

                    /// clean the map for current copy
                    mapper.clear();
                    /// map arguments of block to given args
                    mapper.map(block.getArguments(), args);

                    for (auto &op : block.without_terminator())
                        builder.clone(op, mapper);

                    /// Get the operands from the block given as parameter
                    /// to the copyBlock function
                    auto operands = [&]()
                    {
                        auto term = block.getTerminator();
                        /// if the joining node is a returnBlock (given previously)
                        if (joiningBlock == returnBlock)
                            return mlir::cast<mlir::func::ReturnOp>(term).getOperands();
                        else
                            return mlir::cast<mlir::cf::BranchOp>(term).getDestOperands();
                    }();

                    /// Create the yield values from the previous mapper
                    yieldVals.clear();
                    /// reserve the size
                    /// yieldVals.resize(operands.size());

                    /// get the values from the previous mapper, the values
                    /// must be those from the operands from the terminator
                    for (auto op : operands)
                        yieldVals.emplace_back(mapper.lookupOrDefault(op));
                    /// build the yield operation
                    builder.create<mlir::scf::YieldOp>(loc, yieldVals);
                };

                /// for generating if instructions we need
                /// lambda functions for true and false bodies
                auto trueBodyGenerator = [&](mlir::OpBuilder &builder, mlir::Location loc)
                { return copyBlock(builder, loc, *trueBlock, getOperands(!reverse)); };

                /// now set a variable to know if there's an else block, to know that
                /// we compare the falseBlock, with the joining node, so we would have
                /// the Figure 1 in case they are different, or the Figure 2 in other case
                bool hasElse = (falseBlock != joiningBlock);

                /// Get the resType from terminator operands
                auto resType = [&]()
                {
                    auto term = trueBlock->getTerminator();
                    /// if the next block is a return block
                    /// the terminator is a return statement
                    if (joiningBlock == returnBlock)
                        return mlir::cast<mlir::func::ReturnOp>(term).getOperands().getTypes();
                    else
                    {
                        return mlir::cast<mlir::cf::BranchOp>(term).getDestOperands().getTypes();
                    }
                }();

                /// declaration of the operation to include
                mlir::scf::IfOp ifOp;

                if (hasElse)
                {
                    auto falseBodyGenerator = [&](mlir::OpBuilder &builder, mlir::Location loc)
                    { return copyBlock(builder, loc, *falseBlock, getOperands(reverse)); };

                    /// finally create the ifOp now
                    ifOp = rewriter.create<mlir::scf::IfOp>(loc, cond, trueBodyGenerator, falseBodyGenerator);
                }
                else
                {
                    if (resType.empty())
                        /// If there are not operands in the trueBlock, generate an if
                        /// without else
                        ifOp = rewriter.create<mlir::scf::IfOp>(loc, cond, trueBodyGenerator);
                    else
                    {
                        /// Generate a fake false body
                        auto falseBodyGenerator = [&](mlir::OpBuilder &builder, mlir::Location loc)
                        {
                            auto res = getOperands(reverse);
                            yieldVals.clear();
                            yieldVals.reserve(res.size());
                            for (auto op : res)
                            {
                                yieldVals.emplace_back(mapper.lookupOrDefault(op));
                            }
                            builder.create<mlir::scf::YieldOp>(loc, yieldVals);
                        };

                        /// finally create the if-else operation
                        ifOp =
                            rewriter.create<mlir::scf::IfOp>(loc, cond, trueBodyGenerator, falseBodyGenerator);
                    }
                }

                /// replace operation with a the result from the if operation
                auto *region = op->getParentRegion();
                if (joiningBlock == returnBlock)
                {
                    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                                      ifOp.getResults());
                }
                else
                {
                    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, joiningBlock,
                                                                    ifOp.getResults());
                }

                (void)mlir::simplifyRegions(rewriter, *region);
                return mlir::success();
            }
            return mlir::failure();
        }
    };

    /// @brief Return the defined values in the blocks from a method
    /// @param blocks blocks where to look for the defined values
    /// @param postBlock an optional post block
    /// @return list of defined values
    static llvm::SmallVector<mlir::Value>
    getDefinedValues(llvm::ArrayRef<mlir::Block *> blocks,
                     mlir::Block *postBlock = nullptr)
    {
        llvm::SmallVector<mlir::Value> ret;
        /// no blocks? return
        if (blocks.empty())
            return ret;

        /// Get the region where the basic blocks are
        auto region = blocks.front()->getParent();

        /// check all the uses from the value
        /// and where it's used
        auto checkVal = [&](mlir::Value val)
        {
            for (auto &use : val.getUses())
            {
                /// get the block where the use is
                auto block = region->findAncestorBlockInRegion(*use.getOwner()->getBlock());

                /// check if the value is not used or used out of
                /// the checked regions
                if (!block || !llvm::is_contained(blocks, block))
                {
                    ret.emplace_back(val);
                    return;
                }
            }
        };

        /// study the dominance from the blocks
        mlir::DominanceInfo dom;
        for (auto block : blocks)
        {
            /// in case user gave a post block,
            /// but the block does not dominate it
            /// continue
            if (postBlock && !dom.dominates(block, postBlock))
                continue;

            /// get the arguments from the block
            /// and check if they have to be added
            for (auto arg : block->getArguments())
                checkVal(arg);
            /// get the results from the operations
            /// this must be a definition
            for (auto &op : block->getOperations())
                for (auto res : op.getResults())
                    checkVal(res);
        }

        return ret;
    }

    static std::optional<mlir::Block *>
    tailLoopToWhile(mlir::PatternRewriter &rewriter, mlir::Location loc,
                    mlir::Block *bodyBlock, mlir::ValueRange initArgs)
    {
        /// we need the bodyBlock to work
        assert(bodyBlock);

        /// get the branch instruction from the body of he loop
        auto bodyBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(bodyBlock->getTerminator());

        /// if it's not a conditional branch
        /// leave
        if (!bodyBr)
            return std::nullopt;

        /// if the conditional jump jumps always to
        /// same place, leave too
        if (bodyBr.getTrueDest() == bodyBr.getFalseDest())
            return std::nullopt;

        /// as before, go over false and true, starting by false first
        for (bool reverse : {false, true})
        {
            /// first we take the body block as the true branch destination
            auto bodyBlock1 = reverse ? bodyBr.getFalseDest() : bodyBr.getTrueDest();

            /// now the exit block, in first case, this is the false destination
            auto exitBlock = reverse ? bodyBr.getTrueDest() : bodyBr.getFalseDest();

            /// check that the bodyBlock1 and given bodyBlock
            /// are the same, that means the loop's body is
            /// correct
            if (bodyBlock1 != bodyBlock)
                continue; /// if not okay, try the other way

            /// get the arguments from the body block
            mlir::ValueRange bodyArgs =
                reverse ? bodyBr.getFalseDestOperands() : bodyBr.getTrueDestOperands();
            /// and the arguments from the exit block
            mlir::ValueRange exitArgs =
                reverse ? bodyBr.getTrueDestOperands() : bodyBr.getFalseDestOperands();
            /// get the list of definitions from loop's block
            llvm::SmallVector<mlir::Value> toReplace = getDefinedValues(bodyBlock);

            /// mapping of different variables
            mlir::IRMapping mapping;

            auto beforeBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                                     mlir::ValueRange args)
            {
                /// check given arguments are the same size than loop's body
                /// loop's exit and definition size
                assert(bodyArgs.size() + exitArgs.size() + toReplace.size() ==
                       args.size());

                /// map the values from the body args, the exit args
                /// to values given in args
                mapping.map(bodyArgs, args.take_front(bodyArgs.size()));
                mapping.map(exitArgs, args.take_back(exitArgs.size()));
                mapping.map(bodyBlock->getArguments(), args.take_front(bodyArgs.size()));

                /// clone the operands from the operations
                for (auto &op : bodyBlock->without_terminator())
                    builder.clone(op, mapping);

                /// Get in results the body args and exit args
                /// values
                llvm::SmallVector<mlir::Value> results;
                results.reserve(bodyArgs.size() + exitArgs.size());

                for (mlir::ValueRange ranges : {bodyArgs, exitArgs})
                    for (mlir::Value val : ranges)
                        results.emplace_back(mapping.lookupOrDefault(val));

                /// get the values to replace in results
                for (auto val : toReplace)
                    results.emplace_back(mapping.lookupOrDefault(val));

                /// create a condition from the condition of the body branch
                mlir::Value cond = mapping.lookupOrDefault(bodyBr.getCondition());
                if (reverse)
                {
                    mlir::Value one =
                        builder.create<mlir::arith::ConstantIntOp>(loc, 1, /*width*/ 1);
                    cond = builder.create<mlir::arith::XOrIOp>(loc, cond, one);
                }
                /// create a condition for while
                builder.create<mlir::scf::ConditionOp>(loc, cond, results);
            };

            /// simple lambda for creating a YieldOperation for returning from While
            auto afterBuilder = [](mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ValueRange args)
            {
                builder.create<mlir::scf::YieldOp>(loc, args);
            };

            /// get the type from body arguments and exit arguments
            auto bodyArgsTypes = bodyArgs.getTypes();
            auto exitTypes = exitArgs.getTypes();

            /// value range of the definitions in loop's body
            mlir::ValueRange toReplaceRange(toReplace);
            /// type of values to replace
            auto definedTypes = toReplaceRange.getTypes();

            /// add all the arguments and defined value types
            /// into a vector of types (probably to create
            /// values for all of them
            llvm::SmallVector<mlir::Type> whileTypes(bodyArgsTypes.begin(),
                                                     bodyArgsTypes.end());
            whileTypes.append(exitTypes.begin(), exitTypes.end());
            whileTypes.append(definedTypes.begin(), definedTypes.end());
            /// arguments for while loop (this will be used inside)
            llvm::SmallVector<mlir::Value> whileArgs(initArgs.begin(), initArgs.end());

            mlir::OpBuilder::InsertionGuard g(rewriter);
            /// locations for the body of the while loop
            llvm::SmallVector<mlir::Location> locs(bodyBlock->getNumArguments(),
                                                   rewriter.getUnknownLoc());

            /// create a block for the while loop
            auto newBlock = rewriter.createBlock(bodyBlock->getParent(), {},
                                                 bodyBlock->getArgumentTypes(), locs);

            /// create undef values for each one of the arguments
            /// from the while loop
            for (auto types : {exitTypes, definedTypes})
            {
                for (auto type : types)
                {
                    mlir::Value val = rewriter.create<::mlir::KUNAI::MjolnIR::UndefOp>(loc, type);
                    whileArgs.emplace_back(val);
                }
            }

            /// create a while operation
            /// using the whileArgs created
            /// this probably will be replaced
            /// for correct one
            auto whileOp = rewriter.create<mlir::scf::WhileOp>(
                loc, whileTypes, whileArgs, beforeBuilder, afterBuilder);

            auto results = whileOp.getResults();
            auto bodyResults = results.take_front(bodyArgsTypes.size());
            auto exitResults =
                results.drop_front(bodyArgsTypes.size()).take_front(exitTypes.size());
            auto definedResults = results.take_back(toReplace.size());
            assert(bodyResults.size() == bodyArgs.size());
            assert(exitResults.size() == exitBlock->getNumArguments());
            /// create the branch to go out to the exit block
            rewriter.create<mlir::cf::BranchOp>(loc, exitBlock, exitResults);
            /// replace old values to replace, for new defined results
            for (auto &&[oldVal, newVal] : llvm::zip(toReplace, definedResults))
                rewriter.replaceAllUsesWith(oldVal, newVal);

            if (llvm::hasSingleElement(bodyBlock->getUses()))
                eraseBlocks(rewriter, bodyBlock);

            return newBlock;
        }
        return std::nullopt;
    }

    struct TailLoopToWhile : public mlir::OpRewritePattern<mlir::cf::BranchOp>
    {
        // Set benefit higher than execute_region _passes
        TailLoopToWhile(mlir::MLIRContext *context)
            : mlir::OpRewritePattern<mlir::cf::BranchOp>(context,
                                                         /*benefit*/ 10)
        {
        }

        mlir::LogicalResult
        matchAndRewrite(mlir::cf::BranchOp op,
                        mlir::PatternRewriter &rewriter) const override
        {
            /// get the block where conditional jump goes
            auto bodyBlock = op.getDest();
            /// apply the transformation to obtain a while loop with a condition
            /// and a yield value
            auto res =
                tailLoopToWhile(rewriter, op.getLoc(), bodyBlock, op.getDestOperands());
            if (!res)
                return mlir::failure();
            /// set the destination of the branch operation to
            /// the created block
            rewriter.updateRootInPlace(op, [&]()
                                       { op.setDest(*res); });
            return mlir::success();
        }
    };

    struct TailLoopToWhileCond
        : public mlir::OpRewritePattern<mlir::cf::CondBranchOp>
    {
        // Set benefit higher than execute_region _passes
        TailLoopToWhileCond(mlir::MLIRContext *context)
            : mlir::OpRewritePattern<mlir::cf::CondBranchOp>(context,
                                                             /*benefit*/ 10)
        {
        }

        mlir::LogicalResult
        matchAndRewrite(mlir::cf::CondBranchOp op,
                        mlir::PatternRewriter &rewriter) const override
        {
            /// in case we have a conditional loop instead of a branch
            for (bool reverse : {false, true})
            {
                /// we will follow common approach of going from true destination
                /// to false destination, and viceversa
                auto bodyBlock = reverse ? op.getFalseDest() : op.getTrueDest();
                if (bodyBlock == op->getBlock())
                    continue;
                /// try to apply the transformation
                auto args =
                    reverse ? op.getFalseDestOperands() : op.getTrueDestOperands();
                auto res = tailLoopToWhile(rewriter, op.getLoc(), bodyBlock, args);
                if (!res)
                    continue;
                /// in case we 
                auto newTrueDest = reverse ? op.getTrueDest() : *res;
                auto newFalseDest = reverse ? *res : op.getFalseDest();
                rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
                    op, op.getCondition(), newTrueDest, op.getTrueDestOperands(),
                    newFalseDest, op.getFalseDestOperands());
                return mlir::success();
            }
            return mlir::failure();
        }
    };

    class CfgToScf : public mlir::PassWrapper<CfgToScf, mlir::OperationPass<void>>
    {
    public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CfgToScf);

        /// @brief Get in the DialectRegistry the other registers
        /// @param registry object to registry other dialects to use
        virtual void
        getDependentDialects(mlir::DialectRegistry &registry) const override
        {
            registry.insert<mlir::scf::SCFDialect>();
        }

        void runOnOperation() override
        {
            assert(mlir::isa<mlir::FunctionOpInterface>(getOperation()) &&
                   "Operation provided must implement FunctionOpInterface.");

            auto context = &getContext();

            mlir::RewritePatternSet patterns(context);

            patterns.insert<ScfIfRewriteOneExit>(context);
            patterns.insert<TailLoopToWhile>(context);
            patterns.insert<TailLoopToWhileCond>(context);

            /// Get the canonicalization patterns from ControlFlowDialect
            context->getLoadedDialect<mlir::cf::ControlFlowDialect>()
                ->getCanonicalizationPatterns(patterns);
            /// Different Canonicalization patterns for each different opcode
            mlir::cf::BranchOp::getCanonicalizationPatterns(patterns, context);
            mlir::cf::CondBranchOp::getCanonicalizationPatterns(patterns, context);
            mlir::scf::ExecuteRegionOp::getCanonicalizationPatterns(patterns, context);
            mlir::scf::IfOp::getCanonicalizationPatterns(patterns, context);

            auto op = getOperation();

            mlir::OperationFingerPrint fp(op);
            int maxIters = 10;
            mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

            // Repeat transformations multiple times until they converge.
            // TODO Not clear why it's needed, investigate later.
            for (auto i : llvm::seq(0, maxIters))
            {
                (void)i;
                /// Apply the different patterns to the operations
                (void)mlir::applyPatternsAndFoldGreedily(op, frozenPatterns);
                mlir::OperationFingerPrint newFp(op);
                if (newFp == fp)
                    break;

                fp = newFp;
            }
            /// Walk over the operations, and look for a branch operation
            /// or a conditional branch from the ControlFlow dialect
            /// this means the transformation was incorrect.
            op->walk([&](mlir::Operation *o) -> mlir::WalkResult
                     {
                    if (mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(o)) {
                    o->emitError("Unable to convert CFG to SCF");
                    signalPassFailure();
                    return mlir::WalkResult::interrupt();
                    }
                    return mlir::WalkResult::advance(); });
        }
    };
}

namespace KUNAI
{
    namespace MjolnIR
    {
        std::unique_ptr<mlir::Pass> createMjolnIRCfgToScfgPass()
        {
            return std::make_unique<CfgToScf>();
        }
    }
}