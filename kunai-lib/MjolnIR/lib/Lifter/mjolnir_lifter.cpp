//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file mjolnir_lifter.cpp

#include "Lifter/mjolnir_lifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"

#include <iostream>
#include <unordered_map>

using namespace KUNAI::MjolnIR;

mlir::LogicalResult MjolnIRLifter::declareReg(std::uint32_t reg,
                                              KUNAI::DEX::EncodedMethod *EM,
                                              mlir::Value value)
{
    if (registerTable.count(reg))
        return mlir::failure();
    /// no register created, create it
    registerTable.insert(reg, {value, EM});
    return mlir::success();
}

mlir::Type MjolnIRLifter::get_type(KUNAI::DEX::DVMFundamental *fundamental)
{
    mlir::Type current_type;

    switch (fundamental->get_fundamental_type())
    {
    case KUNAI::DEX::DVMFundamental::BOOLEAN:
        current_type = ::mlir::KUNAI::MjolnIR::DVMBoolType();
        break;
    case KUNAI::DEX::DVMFundamental::BYTE:
        current_type = ::mlir::KUNAI::MjolnIR::DVMByteType();
        break;
    case KUNAI::DEX::DVMFundamental::CHAR:
        current_type = ::mlir::KUNAI::MjolnIR::DVMCharType();
        break;
    case KUNAI::DEX::DVMFundamental::DOUBLE:
        current_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType();
        break;
    case KUNAI::DEX::DVMFundamental::FLOAT:
        current_type = ::mlir::KUNAI::MjolnIR::DVMFloatType();
        break;
    case KUNAI::DEX::DVMFundamental::INT:
        current_type = ::mlir::KUNAI::MjolnIR::DVMIntType();
        break;
    case KUNAI::DEX::DVMFundamental::LONG:
        current_type = ::mlir::KUNAI::MjolnIR::DVMLongType();
        break;
    case KUNAI::DEX::DVMFundamental::SHORT:
        current_type = ::mlir::KUNAI::MjolnIR::DVMShortType();
        break;
    case KUNAI::DEX::DVMFundamental::VOID:
        current_type = ::mlir::KUNAI::MjolnIR::DVMVoidType();
        break;
    }

    return current_type;
}

mlir::Type MjolnIRLifter::get_type(KUNAI::DEX::DVMType *type)
{
    if (type->get_type() == KUNAI::DEX::DVMType::FUNDAMENTAL)
        return get_type(reinterpret_cast<KUNAI::DEX::DVMFundamental *>(type));
    else if (type->get_type() == KUNAI::DEX::DVMType::CLASS)
        throw exceptions::LifterException("MjolnIRLifter::get_type: type CLASS not implemented yet...");
    else if (type->get_type() == KUNAI::DEX::DVMType::ARRAY)
        throw exceptions::LifterException("MjolnIRLIfter::get_type: type ARRAY not implemented yet...");
    else
        throw exceptions::LifterException("MjolnIRLifter::get_type: that type is unknown or I don't know what it is...");
}

llvm::SmallVector<mlir::Type> MjolnIRLifter::gen_prototype(KUNAI::DEX::ProtoID *proto)
{
    llvm::SmallVector<mlir::Type, 4> argTypes;

    /// as much space as parameters
    argTypes.reserve(proto->get_parameters().size());

    /// since we have a vector of parameters
    /// it is easy peasy
    for (auto param : proto->get_parameters())
        argTypes.push_back(get_type(param));

    return argTypes;
}

::mlir::KUNAI::MjolnIR::MethodOp MjolnIRLifter::get_method(KUNAI::DEX::EncodedMethod *encoded_method)
{
    auto method = encoded_method->getMethodID();

    KUNAI::DEX::ProtoID *proto = method->get_proto();
    std::string &name = method->get_name();

    auto method_location = mlir::FileLineColLoc::get(&context, llvm::StringRef(name.c_str()), 0, 0);

    // now let's create a MethodOp, for that we will need first to retrieve
    // the type of the parameters
    auto paramTypes = gen_prototype(proto);

    // now retrieve the return type
    auto retType = get_type(proto->get_return_type());

    // create now the method type
    auto methodType = builder.getFunctionType(paramTypes, {retType});

    auto methodOp = builder.create<::mlir::KUNAI::MjolnIR::MethodOp>(method_location, name, methodType);

    /// declare the register parameters, these are used during the
    /// program
    auto number_of_params = proto->get_parameters().size();

    auto number_of_registers = encoded_method->get_code_item().get_registers_size();

    for (std::uint32_t Param = (number_of_registers-number_of_params+1),    /// starting index of the parameter
        Limit = (static_cast<std::uint32_t>(number_of_registers)+1),        /// limit value for parameters
        Argument = 0;                                                       /// for obtaining parameter by index 0
        Param < Limit;
        ++Param,
        ++Argument)
    {
        auto value = methodOp.getArgument(Argument);
        if(failed(declareReg(Param, encoded_method, value)))
            throw exceptions::LifterException("MjolnIRLifter::get_method: trying to declare register already declared");
    }

    // with the type created, now create the Method
    return methodOp;
}

void MjolnIRLifter::gen_instruction(KUNAI::DEX::Instruction *instr)
{
}

mlir::OwningOpRef<mlir::ModuleOp> MjolnIRLifter::mlirGen(KUNAI::DEX::MethodAnalysis *methodAnalysis)
{
    Module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // create an MLIR function for the prototype
    builder.setInsertionPointToEnd(Module.getBody());

    auto function = get_method(
        std::get<KUNAI::DEX::EncodedMethod *>(methodAnalysis->get_encoded_method()));
    // let's start now the body of the function
    mlir::Block &entryBlock = function.front();

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    return Module;
}