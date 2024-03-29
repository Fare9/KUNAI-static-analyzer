//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.cpp

#include "Dalvik/MjolnIROps.hpp"
#include "Dalvik/MjolnIRDialect.hpp"
#include "Dalvik/MjolnIRTypes.hpp"

// include from MLIR
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
// include from LLVM
#include <llvm/ADT/TypeSwitch.h>

#define GET_OP_CLASSES
#include "Dalvik/MjolnIROps.cpp.inc"

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

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range InvokeOp::getArgOperands() { return getInputs(); }

//===----------------------------------------------------------------------===//
// FallthroughOp
//===----------------------------------------------------------------------===//
void FallthroughOp::setDest(Block *block) { return setSuccessor(block); }

void FallthroughOp::eraseOperand(unsigned index) { (*this)->eraseOperand(index); }
