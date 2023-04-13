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

mlir::LogicalResult MjolnIRLifter::updateReg(std::uint32_t reg, mlir::Value value)
{
    if (!registerTable.count(reg))
        return mlir::failure();
    auto EM = registerTable.lookup(reg).second;

    registerTable.insert(reg, {value, EM});
    return mlir::success();
}

mlir::Type MjolnIRLifter::get_type(KUNAI::DEX::DVMFundamental *fundamental)
{
    mlir::Type current_type;

    switch (fundamental->get_fundamental_type())
    {
    case KUNAI::DEX::DVMFundamental::BOOLEAN:
        current_type = ::mlir::KUNAI::MjolnIR::DVMBoolType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::BYTE:
        current_type = ::mlir::KUNAI::MjolnIR::DVMByteType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::CHAR:
        current_type = ::mlir::KUNAI::MjolnIR::DVMCharType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::DOUBLE:
        current_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::FLOAT:
        current_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::INT:
        current_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::LONG:
        current_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::SHORT:
        current_type = ::mlir::KUNAI::MjolnIR::DVMShortType::get(&context);
        break;
    case KUNAI::DEX::DVMFundamental::VOID:
        current_type = ::mlir::KUNAI::MjolnIR::DVMVoidType::get(&context);
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
    mlir::Type retType = get_type(proto->get_return_type());

    // create now the method type
    auto methodType = builder.getFunctionType(paramTypes, {retType});

    auto methodOp = builder.create<::mlir::KUNAI::MjolnIR::MethodOp>(method_location, name, methodType);

    /// declare the register parameters, these are used during the
    /// program
    auto number_of_params = proto->get_parameters().size();

    auto number_of_registers = encoded_method->get_code_item().get_registers_size();

    for (std::uint32_t Param = (number_of_registers - number_of_params), /// starting index of the parameter
         Limit = (static_cast<std::uint32_t>(number_of_registers)),      /// limit value for parameters
         Argument = 0;                                                   /// for obtaining parameter by index 0
         Param < Limit;
         ++Param,
                       ++Argument)
    {
        auto value = methodOp.getArgument(Argument);
        if (failed(declareReg(Param, encoded_method, value)))
            throw exceptions::LifterException("MjolnIRLifter::get_method: trying to declare register already declared");
    }

    // with the type created, now create the Method
    return methodOp;
}

void MjolnIRLifter::gen_instruction(KUNAI::DEX::Instruction23x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, file_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src1 = instr->get_first_source();
    auto src2 = instr->get_second_source();

    mlir::Type dest_type = nullptr;

    switch (op_code)
    {
    /// Different Add Operations
    case KUNAI::DEX::TYPES::OP_ADD_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_ADD_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_ADD_FLOAT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_ADD_DOUBLE:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AddOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;
    /// Different Sub operations
    case KUNAI::DEX::TYPES::OP_SUB_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_SUB_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_SUB_FLOAT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_SUB_DOUBLE:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::SubOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;
    /// Different Mul operations
    case KUNAI::DEX::TYPES::OP_MUL_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_MUL_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_MUL_FLOAT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_MUL_DOUBLE:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::MulOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;
    /// Different Div operations
    case KUNAI::DEX::TYPES::OP_DIV_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_DIV_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_DIV_FLOAT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_DIV_DOUBLE:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::DivOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;
    /// Different Rem operations
    case KUNAI::DEX::TYPES::OP_REM_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_REM_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_REM_FLOAT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_REM_DOUBLE:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::RemOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    /// All And operations
    case KUNAI::DEX::TYPES::OP_AND_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_AND_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AndOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    /// All Or operations
    case KUNAI::DEX::TYPES::OP_OR_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_OR_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::OrOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    /// All Xor operations
    case KUNAI::DEX::TYPES::OP_XOR_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_XOR_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::XorOp>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    /// All SHL instructions
    case KUNAI::DEX::TYPES::OP_SHL_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_SHL_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shl>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    /// All SHR instructions
    case KUNAI::DEX::TYPES::OP_SHR_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_SHR_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shr>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    /// All USHR instructions
    case KUNAI::DEX::TYPES::OP_USHR_INT:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_USHR_LONG:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::UShr>(
                location,
                dest_type,
                registerTable.lookup(src1).first,
                registerTable.lookup(src2).first);
            if (failed(declareReg(dest, registerTable.lookup(src1).second, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: error redefinition of register");
        }
        break;

    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Opcode from Instruction23x not implemented");
        break;
    }
}

void MjolnIRLifter::gen_instruction(KUNAI::DEX::Instruction11x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, file_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_RETURN:
    case KUNAI::DEX::TYPES::OP_RETURN_WIDE:
    case KUNAI::DEX::TYPES::OP_RETURN_OBJECT:
    {
        auto reg_value = registerTable.lookup(dest).first;

        builder.create<::mlir::KUNAI::MjolnIR::ReturnOp>(
            location,
            reg_value);
    }
    break;

    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction11x not supported");
        break;
    }
}

void MjolnIRLifter::gen_instruction(KUNAI::DEX::Instruction12x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, file_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    mlir::Type dest_type = nullptr;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_ADD_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_ADD_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_ADD_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_ADD_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AddOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_SUB_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_SUB_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_SUB_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_SUB_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::SubOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_MUL_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_MUL_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_MUL_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_MUL_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::MulOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_DIV_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_DIV_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_DIV_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_DIV_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::DivOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_REM_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_REM_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
    case KUNAI::DEX::TYPES::OP_REM_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMFloatType::get(&context);
    case KUNAI::DEX::TYPES::OP_REM_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMDoubleType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::RemOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_AND_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_AND_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::AndOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_OR_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_OR_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::OrOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_XOR_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_XOR_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::XorOp>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_SHL_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_SHL_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shl>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_SHR_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_SHR_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Shr>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    case KUNAI::DEX::TYPES::OP_USHR_INT_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMIntType::get(&context);
    case KUNAI::DEX::TYPES::OP_USHR_LONG_2ADDR:
        if (!dest_type)
            dest_type = ::mlir::KUNAI::MjolnIR::DVMLongType::get(&context);
        {
            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::UShr>(
                location,
                dest_type,
                registerTable.lookup(dest).first,
                registerTable.lookup(src).first);
            if (failed(updateReg(dest, generated_value)))
                throw exceptions::LifterException("MjolnIRLifter::gen_instruction: exception updating register value");
        }
        break;

    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction12x not supported");
    }
}

void MjolnIRLifter::gen_instruction(KUNAI::DEX::Instruction *instr)
{
    switch (instr->get_instruction_type())
    {
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION23X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction23x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION12X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction12x *>(instr));
        break;
    case KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11X:
        gen_instruction(reinterpret_cast<KUNAI::DEX::Instruction11x *>(instr));
        break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: InstructionType not implemented");
    }
}

void MjolnIRLifter::gen_block(KUNAI::DEX::DVMBasicBlock *bb)
{
    for (auto instr : bb->get_instructions())
    {
        try
        {
            gen_instruction(instr);
        }
        catch (const exceptions::LifterException &le)
        {
            if (gen_exception)
                throw le;

            auto location = mlir::FileLineColLoc::get(&context, file_name, instr->get_address(), 0);

            builder.create<::mlir::KUNAI::MjolnIR::Nop>(location);
        }
    }
}

mlir::OwningOpRef<mlir::ModuleOp> MjolnIRLifter::mlirGen(KUNAI::DEX::MethodAnalysis *methodAnalysis)
{
    /// initialize an scope for the registers
    RegisterTableScopeT registerScope(registerTable);

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

    if (file_name.empty())
        file_name = methodAnalysis->get_class_name();

    auto &bbs = methodAnalysis->get_basic_blocks();

    for (auto bb : bbs.get_nodes())
        gen_block(bb);

    return Module;
}