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
    static const constexpr bool debugLoopRestructuring = false;

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

    namespace
    {
        /// @brief Structure represented strongly connected components
        struct SCC
        {
            struct Node
            {
                mlir::SmallVector<mlir::Block *, 4> blocks;
            };
            mlir::SmallVector<Node> nodes;

            /// @brief function for dumping the strongly connected components
            void dump() const
            {
                for (auto &&[i, node] : llvm::enumerate(nodes))
                {
                    llvm::errs() << "scc node " << i << "\n";
                    for (auto b : node.blocks)
                    {
                        llvm::errs() << " block ";
                        b->dump();
                    }
                }
            }
        };

        struct BlockDesc
        {
            enum
            {
                UndefinedIndex = -1
            };
            int index = UndefinedIndex;
            int lowLink = UndefinedIndex;
            bool onStack = false;
        };

        /// @brief define an Edge type for the mlir blocks
        using Edge = std::pair<mlir::Block *, mlir::Block *>;
    } /// namespace

    static void strongconnect(
        mlir::Block *block,
        llvm::SmallDenseMap<mlir::Block *, BlockDesc> &blocks,
        llvm::SmallVectorImpl<mlir::Block *> &stack,
        int &index,
        SCC &scc)
    {
        assert(block);
        auto &desc = blocks[block];
        /// check if the description of the block
        /// was already initialized
        if (desc.index != BlockDesc::UndefinedIndex)
            return;
        /// initialize the description of the block
        desc.index = index;
        desc.lowLink = index;
        ++index;

        desc.onStack = true;
        stack.push_back(block);

        /// get the region where blocks are contained
        auto region = block->getParent();
        for (mlir::Block *successor : block->getSuccessors())
        {
            /// go over the successors, we will create strongly connected components

            if (region != successor->getParent())
                continue;

            auto &successorDesc = blocks[successor];

            bool update = false;
            if (successorDesc.index == BlockDesc::UndefinedIndex) /// if successor has not been processed
            {
                /// Strongly connect components
                strongconnect(successor, blocks, stack, index, scc);
                update = true;
            }
            else if (successorDesc.onStack)
            {
                update = true;
            }

            if (update)
            {
                /// blocks dense map may have been reallocated, retrieve
                /// again the values
                auto &successorDesc1 = blocks[successor];
                auto &desc1 = blocks[block];
                /// for predecessor block, set the low link
                desc1.lowLink = std::min(desc1.lowLink, successorDesc1.index);
            }
        }

        auto &desc1 = blocks[block];
        /// if blocks were connected to the current one
        if (desc1.lowLink != desc1.index)
            return;

        auto &sccNode = scc.nodes.emplace_back();
        mlir::Block *currentBlock = nullptr;
        /// go over the stack of nodes adding it to
        /// to the vector of strongly connected blocks
        do
        {
            assert(!stack.empty());
            currentBlock = stack.pop_back_val();
            blocks[currentBlock].onStack = false;
            sccNode.blocks.emplace_back(currentBlock);
        } while (currentBlock != block);
    }

    /// SCC construction algorithm from
    /// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    static SCC buildSCC(mlir::Region &region)
    {
        SCC scc;

        llvm::SmallDenseMap<mlir::Block *, BlockDesc> blocks;
        llvm::SmallVector<mlir::Block *> stack;
        int index = 0;
        /// strongly connect the blocks from the region
        /// following Tarjan algorithm
        for (auto &block : region)
            strongconnect(&block, blocks, stack, index, scc);

        return scc;
    }

    /// @brief From a terminator instruction retrieve its operands
    /// in case it's a conditional terminator, get the one from
    /// target provided
    /// @param term instruction to retrieve arguments
    /// @param target target in case a conditional branch is provided.
    /// @return
    static mlir::ValueRange getTerminatorArgs(mlir::Operation *term, mlir::Block *target)
    {
        assert(term);
        assert(target);

        if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(term)) /// in case it is a branch operation
        {
            assert(target == br.getDest());
            return br.getDestOperands();
        }

        if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(term))
        {
            assert(target == condBr.getTrueDest() || target == condBr.getFalseDest());

            return target == condBr.getTrueDest() ? condBr.getTrueDestOperands() : condBr.getFalseDestOperands();
        }

        llvm_unreachable("getTerminatorArgs: terminator provided not supported");
    }

    /// @brief From an edge object (Node->Node) retrieve
    /// the arguments from the terminator instruction.
    /// @param edge edge to retrieve the arguments
    /// @return arguments from edge
    static mlir::ValueRange getEdgeArgs(Edge edge)
    {
        auto term = edge.first->getTerminator();
        auto args = getTerminatorArgs(term, edge.second);
        assert(args.size() == edge.second->getNumArguments());
        return args;
    }

    /// @brief Replace the destination from an edge for a new destination
    /// @param rewriter rewriter from MLIR to create a new instruction
    /// @param edge edge where to modify the destination
    /// @param newDest new destination to set
    /// @param newArgs new arguments to set
    static void replaceEdgeDest(mlir::PatternRewriter &rewriter,
                                Edge edge,
                                mlir::Block *newDest,
                                mlir::ValueRange newArgs)
    {
        /// check that the new arguments and the number of
        /// arguments of the new destination have same size
        assert(newDest->getNumArguments() == newArgs.size());

        auto term = edge.first->getTerminator();
        /// keep the insertion point
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(term); /// set the insertion point in the terminator instruction

        /// replace a branch instruction
        if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(term)) /// is a BranchOp?
        {
            assert(edge.second == br.getDest());
            /// replace previous branch, with a new one pointing to a new destination
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(br, newDest, newArgs);
            return;
        }

        /// replace a conditional branch instruction
        if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(term))
        {
            mlir::Block *trueDest = condBr.getTrueDest();
            mlir::ValueRange trueArgs = condBr.getTrueDestOperands();
            mlir::Block *falseDest = condBr.getFalseDest();
            mlir::ValueRange falseArgs = condBr.getFalseDestOperands();

            if (edge.second == trueDest) /// replace the true destination
            {
                trueDest = newDest;
                trueArgs = newArgs;
            }
            if (edge.second == falseDest)
            {
                falseDest = newDest;
                falseArgs = falseArgs;
            }

            auto cond = condBr.getCondition();
            rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(condBr, cond, trueDest, trueArgs, falseDest, falseArgs);
            return;
        }

        llvm_unreachable("replaceEdgeDest: unsupported terminator");
    }

    static void addTypesFromEdges(llvm::ArrayRef<Edge> edges, llvm::SmallVectorImpl<mlir::Type> &ret)
    {
        for (auto edge : edges)
        {
            auto edgeArgs = edge.second->getArgumentTypes();
            ret.append(edgeArgs.begin(), edgeArgs.end());
        }
    }

    static void generateMultiplexedBranches(mlir::PatternRewriter &rewriter,
                                            mlir::Location loc,
                                            mlir::Block *srcBlock,
                                            mlir::ValueRange multiplexArgs,
                                            mlir::ValueRange srcArgs,
                                            llvm::ArrayRef<Edge> edges)
    {
        assert(srcBlock);
        assert(!edges.empty());

        /// store the point where we are inserting data
        mlir::OpBuilder::InsertionGuard g(rewriter);

        auto region = srcBlock->getParent();
        auto numMultiplexVars = edges.size() - 1;
        assert(multiplexArgs.size() == numMultiplexVars);

        /// only two nodes
        if (edges.size() == 1)
        {
            rewriter.setInsertionPointToEnd(srcBlock);
            auto dst = edges.front().second;
            assert(dst->getNumArguments() == srcArgs.size());
            rewriter.create<mlir::cf::BranchOp>(loc, dst, srcArgs);
            return;
        }

        mlir::Block *currentBlock = srcBlock;
        for (auto &&[i, edge] : llvm::enumerate(edges.drop_back()))
        {
            /// get destination of the edge
            mlir::Block *dst = edge.second;
            auto numArgs = dst->getNumArguments();
            auto args = srcArgs.take_front(numArgs);
            /// check the args from the source and destination
            /// are of same size
            assert(numArgs == args.size());

            rewriter.setInsertionPointToEnd(currentBlock);

            auto cond = multiplexArgs[i];

            if (i == (numMultiplexVars - 1))
            {
                auto lastEdge = edges.back();
                auto lastDst = lastEdge.second;
                auto falseArgs = srcArgs.drop_front(numArgs);
                assert(lastDst->getNumArguments() == falseArgs.size());
                rewriter.create<mlir::cf::CondBranchOp>(loc, cond, dst, args, lastDst,
                                                        falseArgs);
            }
            else
            {
                auto nextBlock = [&]() -> mlir::Block *
                {
                    mlir::OpBuilder::InsertionGuard g(rewriter);
                    return rewriter.createBlock(region);
                }();
                assert(nextBlock->getNumArguments() == 0);
                rewriter.create<mlir::cf::CondBranchOp>(loc, cond, dst, args, nextBlock,
                                                        mlir::ValueRange{});
                currentBlock = nextBlock;
            }
            srcArgs = srcArgs.drop_front(numArgs);
        }
    }

    static void initMultiplexConds(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, size_t currentBlock,
                                   size_t numBlocks,
                                   llvm::SmallVectorImpl<mlir::Value> &res)
    {
        assert(numBlocks > 0);
        assert(currentBlock < numBlocks);
        auto boolType = rewriter.getI1Type();
        auto trueVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, boolType);
        auto falseVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, boolType);
        for (auto j : llvm::seq<size_t>(0, numBlocks - 1))
        {
            auto val = (j == currentBlock ? trueVal : falseVal);
            res.emplace_back(val);
        }
    }

    static void initUndefMultiplexConds(mlir::PatternRewriter &rewriter,
                                        mlir::Location loc, size_t numBlocks,
                                        llvm::SmallVectorImpl<mlir::Value> &res)
    {
        assert(numBlocks > 0);
        auto boolType = rewriter.getI1Type();
        auto undefVal = rewriter.create<mlir::KUNAI::MjolnIR::UndefOp>(loc, boolType);
        for (auto j : llvm::seq<size_t>(0, numBlocks - 1))
        {
            (void)j;
            res.emplace_back(undefVal);
        }
    }

    static void initMultiplexVars(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, size_t currentBlock,
                                  llvm::ArrayRef<Edge> edges,
                                  llvm::SmallVectorImpl<mlir::Value> &res)
    {
        assert(currentBlock < edges.size());
        for (auto &&[j, edge] : llvm::enumerate(edges))
        {
            mlir::ValueRange args = getEdgeArgs(edge);
            if (j == currentBlock)
            {
                res.append(args.begin(), args.end());
            }
            else
            {
                for (auto type : args.getTypes())
                {
                    mlir::Value init = rewriter.create<mlir::KUNAI::MjolnIR::UndefOp>(loc, type);
                    res.emplace_back(init);
                }
            }
        }
    }

    static void initUndefMultiplexVars(mlir::PatternRewriter &rewriter,
                                       mlir::Location loc,
                                       llvm::ArrayRef<Edge> edges,
                                       llvm::SmallVectorImpl<mlir::Value> &res)
    {
        for (auto &&[j, edge] : llvm::enumerate(edges))
        {
            for (auto type : edge.second->getArgumentTypes())
            {
                mlir::Value init = rewriter.create<mlir::KUNAI::MjolnIR::UndefOp>(loc, type);
                res.emplace_back(init);
            }
        }
    }

    /// @brief Check if a loop is structured, check the edges are correct.
    /// @param inEdges input edges from the loop
    /// @param outEdges output edges from the loop
    /// @param repEdges these must be those that point to itself
    /// @return if loop is structured
    static bool isStructuredLoop(llvm::ArrayRef<Edge> inEdges,
                                 llvm::ArrayRef<Edge> outEdges,
                                 llvm::ArrayRef<Edge> repEdges)
    {
        if (inEdges.empty())
            return false;

        if (outEdges.size() != 1)
            return false;

        auto outBlock = outEdges.front().first;

        auto inBlock = inEdges.front().second;

        for (auto edge : inEdges.drop_front())
        {
            if (edge.second != inBlock) /// check if point to the correct block
                return false;
        }

        if (outBlock->getNumSuccessors() != 2) /// check the out block of the loop
                                               /// has only two sucessors (out and begin of loop)
            return false;

        /// get sucessors from out block of the loop
        auto succ1 = outBlock->getSuccessor(0);
        auto succ2 = outBlock->getSuccessor(1);
        /// check successors do not point to same block
        return (succ1 == inBlock && succ2 != inBlock) ||
               (succ2 == inBlock && succ1 != inBlock);
    }

    static void visitBlock(mlir::Block *block,
                           mlir::Block *begin,
                           mlir::Block *end,
                           llvm::SmallSetVector<mlir::Block *, 8> &blocks)
    {
        assert(block);
        assert(begin);
        assert(end);

        /// block cannot be begin nor end
        if (block == begin || block == end)
            return;
        /// if block is already in set of visited blocks...
        if (blocks.count(block))
            return;

        /// visit the block, and continue
        blocks.insert(block);
        for (auto successor : block->getSuccessors())
            visitBlock(successor, begin, end, blocks);
    }

    static auto collectBlocks(mlir::Block *begin, mlir::Block *end)
    {
        assert(begin);
        assert(end);

        llvm::SmallSetVector<mlir::Block *, 8> blocks;
        for (auto successor : begin->getSuccessors())
            visitBlock(successor, begin, end, blocks);

        return blocks.takeVector();
    }

    static mlir::Block *wrapIntoRegion(mlir::PatternRewriter &rewriter,
                                       mlir::Block *entryBlock,
                                       mlir::Block *exitBlock)
    {
        assert(entryBlock);
        assert(exitBlock);
        assert(entryBlock->getParent() == exitBlock->getParent());
        mlir::OpBuilder::InsertionGuard g(rewriter);

        auto region = entryBlock->getParent();
        auto loc = rewriter.getUnknownLoc();
        llvm::SmallVector<mlir::Location> locs(entryBlock->getNumArguments(), loc);

        /// @brief lambda for creating a block
        auto createBlock = [&](mlir::TypeRange types = std::nullopt) -> mlir::Block *
        {
            locs.resize(types.size(), loc);
            return rewriter.createBlock(region, {}, types, locs);
        };

        llvm::SmallVector<mlir::Block *> cachedPredecessors;
        mlir::IRMapping cachedMapping;

        /// @brief Lambda for making predecessors of a block, point to a new block
        auto updatePredecessors = [&](mlir::Block *block, mlir::Block *newBlock)
        {
            assert(block);
            assert(newBlock);
            assert(block->getArgumentTypes() == newBlock->getArgumentTypes());

            cachedMapping.clear();
            cachedMapping.map(block, newBlock);
            auto preds = block->getPredecessors();
            cachedPredecessors.clear();
            cachedPredecessors.assign(preds.begin(), preds.end());
            /// make every predecessor points to the new block
            for (auto predecessor : cachedPredecessors)
            {
                auto term = predecessor->getTerminator();
                rewriter.setInsertionPoint(term);
                /// deep copy of terminal instruction
                /// changed the mapped operands
                rewriter.clone(*term, cachedMapping);
                rewriter.eraseOp(term);
            }
        };

        /// create an empty block
        auto newEntryBlock = createBlock();
        /// predecessor block
        auto preBlock = createBlock(entryBlock->getArgumentTypes());
        /// branch operation to just created block
        rewriter.create<mlir::cf::BranchOp>(loc, newEntryBlock);

        /// update the predecessors of entryBlock to point to preBlock
        updatePredecessors(entryBlock, preBlock);
        /// entryBlock to -> newEntryBlock
        rewriter.mergeBlocks(entryBlock, newEntryBlock, preBlock->getArguments());

        /// create new exit block for the loop
        auto newExitBlock = createBlock(exitBlock->getArgumentTypes());

        /// we will merge the exit block into the new exit block
        /// so clone the exit terminator into postBlock, and erase
        /// current one
        auto exitTerm = exitBlock->getTerminator();
        auto postBlock = createBlock();
        rewriter.clone(*exitTerm);
        rewriter.eraseOp(exitTerm);
        /// now replace the exit block
        updatePredecessors(exitBlock, newExitBlock);
        rewriter.mergeBlocks(exitBlock, newExitBlock, newExitBlock->getArguments());

        /// create a terminator to the post block
        rewriter.setInsertionPointToEnd(newExitBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, postBlock);

        auto blocks = collectBlocks(preBlock, postBlock);

        auto definedValues = getDefinedValues(blocks, postBlock);

        mlir::ValueRange definedValuesRange(definedValues);
        auto newBlock = createBlock();
        auto regionOp = rewriter.create<mlir::scf::ExecuteRegionOp>(
            loc, definedValuesRange.getTypes());

        rewriter.setInsertionPoint(newExitBlock->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(newExitBlock->getTerminator(),
                                                        definedValues);

        for (auto &&[oldVal, newVal] :
             llvm::zip(definedValues, regionOp->getResults()))
        {
            for (auto &use : llvm::make_early_inc_range(oldVal.getUses()))
            {
                auto owner = use.getOwner();
                auto block = region->findAncestorBlockInRegion(*owner->getBlock());
                if (block && llvm::is_contained(blocks, block))
                    continue;

                mlir::Value val = newVal;
                rewriter.updateRootInPlace(owner, [&]()
                                           { use.set(val); });
            }
        }

        auto &regionOpRegion = regionOp.getRegion();
        auto dummyBlock = rewriter.createBlock(&regionOpRegion);
        for (auto block : blocks)
            block->moveBefore(dummyBlock);

        rewriter.eraseBlock(dummyBlock);

        rewriter.mergeBlocks(postBlock, newBlock);

        rewriter.eraseOp(preBlock->getTerminator());
        rewriter.mergeBlocks(newBlock, preBlock);

        return preBlock;
    }

    static void buildEdges(llvm::ArrayRef<mlir::Block *> blocks,
                           llvm::SmallVectorImpl<Edge> &inEdges,
                           llvm::SmallVectorImpl<Edge> &outEdges,
                           llvm::SmallVectorImpl<Edge> &repetitionEdges)
    {
        llvm::SmallDenseSet<mlir::Block *> blocksSet(blocks.begin(), blocks.end());

        auto region = blocks.front()->getParent();

        /// @brief check if block is a strongly connected block
        auto isInSCC = [&](mlir::Block *block)
        {
            assert(block);
            return blocksSet.count(block) != 0;
        };

        for (auto block : blocks)
        {
            bool isInput = false;

            /// generate the inEdges of the loop
            for (auto predecessor : block->getPredecessors())
            {
                /// predecessor block is not in same region
                if (predecessor->getParent() != region)
                    continue;

                if (!isInSCC(predecessor))
                {
                    inEdges.emplace_back(predecessor, block);
                    isInput = true;
                }
            }

            /// generate the outEdges of the loop
            for (auto successor : block->getSuccessors())
            {
                if (successor->getParent() != region)
                    continue;

                if (!isInSCC(successor))
                    outEdges.emplace_back(block, successor);
            }

            /// generate the repetitionEdges
            if (isInput)
            {
                for (auto predecessor : block->getPredecessors())
                {
                    if (predecessor->getParent() != region)
                        continue;

                    if (isInSCC(predecessor))
                        repetitionEdges.emplace_back(predecessor, block);
                }
            }
        }

        if (debugLoopRestructuring)
        {
            auto printEdges = [](auto &edges, llvm::StringRef name)
            {
                llvm::errs() << name << " edges begin\n";
                for (auto e : edges)
                {
                    llvm::errs() << " edge\n";
                    e.first->dump();
                    e.second->dump();
                }
                llvm::errs() << name << " edges end\n";
            };
            printEdges(inEdges, "inEdges");
            printEdges(outEdges, "outEdges");
            printEdges(repetitionEdges, "repetitionEdges");
        }
    }

    /// @brief Main algorithm from paper https://dl.acm.org/doi/pdf/10.1145/2693261
    /// implemented in Numba, restructure loop into tail-controlled form according
    /// to algorithm described in the paper.
    /// @param rewriter rewriter from MLIR
    /// @param node strongly connected components
    /// @return true if any modification was done to the IR
    static bool restructureLoop(mlir::PatternRewriter &rewriter, SCC::Node &node)
    {
        assert(!node.blocks.empty());

        if (node.blocks.size() == 1)
            return false;

        auto &blocks = node.blocks;
        auto region = blocks.front()->getParent();

        /// create edges of blocks
        llvm::SmallVector<Edge> inEdges;
        llvm::SmallVector<Edge> outEdges;
        llvm::SmallVector<Edge> repetitionEdges;
        buildEdges(blocks, inEdges, outEdges, repetitionEdges);

        if (inEdges.empty())
            return false;

        llvm::SmallVector<Edge> multiplexEdges(inEdges.begin(), inEdges.end());
        multiplexEdges.append(repetitionEdges.begin(), repetitionEdges.end());
        assert(!multiplexEdges.empty());

        // Check if we are already in structured form.
        if (isStructuredLoop(inEdges, outEdges, repetitionEdges))
            return false;

        auto boolType = rewriter.getI1Type();
        auto numInMultiplexVars = multiplexEdges.size() - 1;
        mlir::Block *multiplexEntryBlock = nullptr;
        auto loc = rewriter.getUnknownLoc();
        auto createBlock = [&](mlir::TypeRange types =
                                   std::nullopt) -> mlir::Block *
        {
            llvm::SmallVector<mlir::Location> locs(types.size(), loc);
            return rewriter.createBlock(region, {}, types, locs);
        };

        {
            llvm::SmallVector<mlir::Type> entryBlockTypes(numInMultiplexVars, boolType);
            addTypesFromEdges(multiplexEdges, entryBlockTypes);
            multiplexEntryBlock = createBlock(entryBlockTypes);
            mlir::ValueRange blockArgs = multiplexEntryBlock->getArguments();
            generateMultiplexedBranches(rewriter, loc, multiplexEntryBlock,
                                        blockArgs.take_front(numInMultiplexVars),
                                        blockArgs.drop_front(numInMultiplexVars),
                                        multiplexEdges);
        }

        mlir::ValueRange repMultiplexOutVars;
        mlir::ValueRange exitArgs;
        mlir::Block *repBlock = nullptr;
        mlir::Block *exitBlock = nullptr;
        auto numOutMultiplexVars = repetitionEdges.size() + outEdges.size() - 2;
        {
            llvm::SmallVector<mlir::Type> repBlockTypes(numOutMultiplexVars + 1,
                                                        boolType);
            auto prevSize = repBlockTypes.size();
            addTypesFromEdges(repetitionEdges, repBlockTypes);
            auto numRepArgs = repBlockTypes.size() - prevSize;

            addTypesFromEdges(outEdges, repBlockTypes);

            repBlock = createBlock(repBlockTypes);
            exitBlock = createBlock();

            mlir::Value cond = repBlock->getArgument(0);
            auto repBlockArgs =
                repBlock->getArguments().drop_front(numOutMultiplexVars + 1);
            auto repMultiplexVars =
                repBlock->getArguments().drop_front().take_front(numOutMultiplexVars);
            auto repMultiplexRepVars =
                repMultiplexVars.take_front(repetitionEdges.size() - 1);
            repMultiplexOutVars = repMultiplexVars.take_back(outEdges.size() - 1);

            {
                rewriter.setInsertionPointToStart(repBlock);
                mlir::Value falseVal =
                    rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, boolType);
                llvm::SmallVector<mlir::Value> multiplexArgs(inEdges.size(), falseVal);
                multiplexArgs.append(repMultiplexRepVars.begin(),
                                     repMultiplexRepVars.end());

                initUndefMultiplexVars(rewriter, loc, inEdges, multiplexArgs);
                mlir::ValueRange repetitionArgs = repBlockArgs.take_front(numRepArgs);
                multiplexArgs.append(repetitionArgs.begin(), repetitionArgs.end());

                assert(multiplexEntryBlock->getNumArguments() == multiplexArgs.size());
                rewriter.create<mlir::cf::CondBranchOp>(loc, cond, multiplexEntryBlock,
                                                        multiplexArgs, exitBlock,
                                                        mlir::ValueRange{});
            }

            exitArgs = repBlockArgs.drop_front(numRepArgs);

            llvm::SmallVector<mlir::Value> branchArgs;
            llvm::SmallVector<mlir::Block *> toReplace;

            toReplace.clear();
            for (auto &&[i, inEdge] : llvm::enumerate(inEdges))
            {
                auto entryBlock = createBlock();
                rewriter.setInsertionPointToStart(entryBlock);
                branchArgs.clear();
                initMultiplexConds(rewriter, loc, i, multiplexEdges.size(), branchArgs);
                initMultiplexVars(rewriter, loc, i, inEdges, branchArgs);
                initUndefMultiplexVars(rewriter, loc, repetitionEdges, branchArgs);

                assert(multiplexEntryBlock->getNumArguments() == branchArgs.size());
                rewriter.create<mlir::cf::BranchOp>(loc, multiplexEntryBlock, branchArgs);
                toReplace.emplace_back(entryBlock);
            }
            for (auto &&[i, edge] : llvm::enumerate(inEdges))
                replaceEdgeDest(rewriter, edge, toReplace[i], {});

            toReplace.clear();
            for (auto &&[i, repEdge] : llvm::enumerate(repetitionEdges))
            {
                auto preRepBlock = createBlock();
                rewriter.setInsertionPointToStart(preRepBlock);
                mlir::Value trueVal =
                    rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, boolType);

                branchArgs.clear();
                branchArgs.emplace_back(trueVal);

                initMultiplexConds(rewriter, loc, i, repetitionEdges.size(), branchArgs);
                initUndefMultiplexConds(rewriter, loc, outEdges.size(), branchArgs);

                initMultiplexVars(rewriter, loc, i, repetitionEdges, branchArgs);
                initUndefMultiplexVars(rewriter, loc, outEdges, branchArgs);

                assert(branchArgs.size() == repBlock->getNumArguments());
                rewriter.create<mlir::cf::BranchOp>(loc, repBlock, branchArgs);
                toReplace.emplace_back(preRepBlock);
            }
            for (auto &&[i, edge] : llvm::enumerate(repetitionEdges))
                replaceEdgeDest(rewriter, edge, toReplace[i], {});

            toReplace.clear();
            for (auto &&[i, outEdge] : llvm::enumerate(outEdges))
            {
                auto preRepBlock = createBlock();
                rewriter.setInsertionPointToStart(preRepBlock);
                mlir::Value falseVal =
                    rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, boolType);

                branchArgs.clear();
                branchArgs.emplace_back(falseVal);

                initUndefMultiplexConds(rewriter, loc, repetitionEdges.size(),
                                        branchArgs);
                initMultiplexConds(rewriter, loc, i, outEdges.size(), branchArgs);

                initUndefMultiplexVars(rewriter, loc, repetitionEdges, branchArgs);
                initMultiplexVars(rewriter, loc, i, outEdges, branchArgs);

                assert(branchArgs.size() == repBlock->getNumArguments());
                rewriter.create<mlir::cf::BranchOp>(loc, repBlock, branchArgs);
                toReplace.emplace_back(preRepBlock);
            }
            for (auto &&[i, edge] : llvm::enumerate(outEdges))
                replaceEdgeDest(rewriter, edge, toReplace[i], {});
        }

        generateMultiplexedBranches(rewriter, loc, exitBlock, repMultiplexOutVars,
                                    exitArgs, outEdges);

        auto resultingBlock = wrapIntoRegion(rewriter, multiplexEntryBlock, repBlock);

        // Invoke TailLoopToWhile directly, so it will run before region inlining.
        for (auto predBlock : resultingBlock->getPredecessors())
        {
            auto root = mlir::dyn_cast<mlir::cf::BranchOp>(predBlock->getTerminator());
            if (!root)
                continue;

            rewriter.setInsertionPoint(root);
            auto res =
                TailLoopToWhile(rewriter.getContext()).matchAndRewrite(root, rewriter);
            if (mlir::succeeded(res))
                break;
        }

        return true;
    }

    static bool isEntryBlock(mlir::Block &block)
    {
        auto region = block.getParent();
        return &(region->front()) == &block;
    }

    static mlir::LogicalResult runLoopRestructuring(mlir::PatternRewriter &rewriter,
                                                    mlir::Region &region)
    {
        auto scc = buildSCC(region);

        if (debugLoopRestructuring)
            scc.dump();

        bool changed = false;
        for (auto &node : scc.nodes)
            changed = restructureLoop(rewriter, node) || changed;

        return mlir::success(changed);
    }

    struct LoopRestructuringBr : public mlir::OpRewritePattern<mlir::cf::BranchOp>
    {
        // Set low benefit so all simplifications will run first
        LoopRestructuringBr(mlir::MLIRContext *context) : mlir::OpRewritePattern<mlir::cf::BranchOp>(context, /*benefit=*/0)
        {
        }

        mlir::LogicalResult matchAndRewrite(mlir::cf::BranchOp op, mlir::PatternRewriter &rewriter) const override
        {
            auto block = op->getBlock();
            if (!isEntryBlock(*block))
                return mlir::failure();
            return runLoopRestructuring(rewriter, *block->getParent());
        }
    };

    struct LoopRestructuringCondBr
        : public mlir::OpRewritePattern<mlir::cf::CondBranchOp>
    {
        // Set low benefit, so all if simplifications will run first.
        LoopRestructuringCondBr(mlir::MLIRContext *context)
            : mlir::OpRewritePattern<mlir::cf::CondBranchOp>(context,
                                                             /*benefit*/ 0)
        {
        }

        mlir::LogicalResult
        matchAndRewrite(mlir::cf::CondBranchOp op,
                        mlir::PatternRewriter &rewriter) const override
        {
            auto block = op->getBlock();
            if (!isEntryBlock(*block))
                return mlir::failure();

            return runLoopRestructuring(rewriter, *block->getParent());
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

            patterns.insert<
                ScfIfRewriteOneExit,
                LoopRestructuringBr,
                LoopRestructuringCondBr,
                TailLoopToWhile,
                TailLoopToWhileCond>(context);

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