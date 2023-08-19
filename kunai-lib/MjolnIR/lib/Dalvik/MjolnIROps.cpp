//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.cpp

#include "MjolnIR/Dalvik/MjolnIROps.hpp"
#include "MjolnIR/Dalvik/MjolnIRDialect.hpp"
#include "MjolnIR/Dalvik/MjolnIRTypes.hpp"

// include from MLIR
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>

// include from LLVM
#include <llvm/ADT/TypeSwitch.h>

#include "MjolnIR/Dalvik/MjolnIREnums.cpp.inc"

#define GET_OP_CLASSES
#include "MjolnIR/Dalvik/MjolnIROps.cpp.inc"

using namespace mlir;
using namespace ::mlir::KUNAI::MjolnIR;

/***
 * Following the example from the Toy language from MLIR webpage
 * we will provide here some useful methods for managing parsing,
 * printing, and build constructors
 */

/// @brief Parser for binary operation and functions
/// @param parser parser object
/// @param result result
/// @return
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result)
{
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (FunctionType funcType = type.dyn_cast<FunctionType>())
    {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                   result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op)
{
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all of the types are the same, print the type directly
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type)
                     { return type == resultType; }))
    {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

void MethodOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     llvm::StringRef name, mlir::FunctionType type,
                     llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the stateof our MethodOp, and create an entry block
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult MethodOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result)
{
    auto buildFuncType =
        [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string &)
    { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void MethodOp::print(mlir::OpAsmPrinter &p)
{
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Returns the region on the function operation that is callable.
mlir::Region *MethodOp::getCallableRegion() { return &getBody(); }

/// Returns results types that callable region produces when executed
llvm::ArrayRef<mlir::Type> MethodOp::getCallableResults()
{
    return getFunctionType().getResults();
}

mlir::ArrayAttr MethodOp::getCallableArgAttrs()
{
    return mlir::ArrayAttr();
}

mlir::ArrayAttr MethodOp::getCallableResAttrs()
{
    return mlir::ArrayAttr();
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

void InvokeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     StringRef callee, ArrayRef<mlir::Value> arguments, MethodOp &method)
{
    state.addTypes(method.getResultTypes());
    state.addOperands(arguments);
    state.addAttribute("callee",
                       mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable InvokeOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for this operation.
void InvokeOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range InvokeOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// FallthroughOp
//===----------------------------------------------------------------------===//

/// @brief Set the destination block of the fallthrough
/// @param block destination block
void FallthroughOp::setDest(Block *block) { return setSuccessor(block); }

/// @brief Erase an operand by its index
void FallthroughOp::eraseOperand(unsigned index) { (*this)->eraseOperand(index); }

/// @brief For the BranchOpInterface get the sucessor operands from index
/// @param index index to retrieve the sucessor operands
/// @return sucessor operands
SuccessorOperands FallthroughOp::getSuccessorOperands(unsigned index)
{
    assert(index == 0 && "invalid successor index");
    return SuccessorOperands(getDestOperandsMutable());
}

/// @brief For the BranchOpInterface same as before
/// @param  attributes not used
/// @return destination block
Block *FallthroughOp::getSuccessorForOperands(ArrayRef<Attribute>)
{
    return getDest();
}

//===----------------------------------------------------------------------===//
// UndefOp - From numba-mlir project
//===----------------------------------------------------------------------===//

namespace {
struct MergeUndefs : public mlir::OpRewritePattern<KUNAI::MjolnIR::UndefOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(KUNAI::MjolnIR::UndefOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<mlir::FunctionOpInterface>();
    if (!parent)
      return mlir::failure();

    auto &block = parent.front();

    auto type = op.getType();
    auto insertionPoint = [&]() -> mlir::Operation * {
      for (auto &op : block.without_terminator()) {
        if (op.hasTrait<mlir::OpTrait::ConstantLike>())
          continue;

        auto undef = mlir::dyn_cast<KUNAI::MjolnIR::UndefOp>(op);
        if (undef && undef.getType() != type)
          continue;

        return &op;
      }
      return block.getTerminator();
    }();

    if (insertionPoint == op)
      return mlir::failure();

    auto existingUndef = mlir::dyn_cast<KUNAI::MjolnIR::UndefOp>(insertionPoint);
    if (!existingUndef) {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(insertionPoint);
      existingUndef = rewriter.create<KUNAI::MjolnIR::UndefOp>(op.getLoc(), type);
    }

    rewriter.replaceOp(op, existingUndef.getResult());
    return mlir::success();
  }
};

struct SelectOfUndef : public mlir::OpRewritePattern<mlir::arith::SelectOp> {
  // Higher benefit than upstream select patterns
  SelectOfUndef(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::arith::SelectOp>(context, /*benefit*/ 10) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value result;
    if (op.getTrueValue().getDefiningOp<KUNAI::MjolnIR::UndefOp>()) {
      result = op.getFalseValue();
    } else if (op.getFalseValue().getDefiningOp<KUNAI::MjolnIR::UndefOp>()) {
      result = op.getTrueValue();
    } else {
      return mlir::failure();
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ReplaceUndefUse : public mlir::OpRewritePattern<KUNAI::MjolnIR::UndefOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(KUNAI::MjolnIR::UndefOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      auto owner = use.getOwner();
      if (!mlir::isa<mlir::arith::ArithDialect, mlir::math::MathDialect>(
              owner->getDialect()))
        continue;

      if (owner->getNumOperands() != 1 || owner->getNumResults() != 1)
        continue;

      auto resType = owner->getResult(0).getType();
      rewriter.replaceOpWithNewOp<KUNAI::MjolnIR::UndefOp>(owner, resType);
      changed = true;
    }
    return mlir::success(changed);
  }
};
} // namespace

void UndefOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          mlir::MLIRContext *context) {
  results.insert<MergeUndefs, SelectOfUndef, ReplaceUndefUse>(context);
}