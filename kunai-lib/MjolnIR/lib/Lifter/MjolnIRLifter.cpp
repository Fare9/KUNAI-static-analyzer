//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRLifter.cpp
#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"

#include <type_traits>
#include <algorithm>

using namespace KUNAI::MjolnIR;

/// unnamed namespace
namespace
{
    /// Code from: https://github.com/J-MR-T/blc/blob/50da676f8c3fa965d5c877534cb068bdfc95dcf2/src/mlir.cpp#L163-L187
    inline void set_block_args_for_predecessor(mlir::Block *blockArgParent, mlir::Block *pred, mlir::SmallVector<mlir::Value, 4> &jmpParameters)
    {
        auto term = pred->getTerminator();

        assert(term && "Predecessor block doesn't have terminator instruction");

        /// lambda for getting the block of parameters
        auto successorOperands = [&]() -> mlir::SuccessorOperands
        {
            if (auto branchInterface = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(term))
            {
                for (auto [index, predSucc] : llvm::enumerate(branchInterface->getSuccessors()))
                {
                    if (predSucc == blockArgParent)
                        return branchInterface.getSuccessorOperands(index);
                }
            }

            assert(false && "Sucessor not found");
        }();

        for (auto setTo : jmpParameters)
            successorOperands.append(setTo);
    }

    /// @brief MjolnIR only has F32 (float) and F64 (double) floats,
    /// return if it's a float
    /// @param type type to check
    /// @return if type is floating point
    bool is_float(mlir::Type type)
    {
        return (type.isF32() || type.isF64());
    }

    /// @brief MjolnIR has different Integer types, check if it's one of them
    /// @param type type to check
    /// @return if type is integer
    bool is_integer(mlir::Type type)
    {
        return (
            type.isInteger(8) ||
            type.isInteger(16) ||
            type.isInteger(32) ||
            type.isInteger(64));
    }
}

void Lifter::init()
{
    context.getOrLoadDialect<::mlir::KUNAI::MjolnIR::MjolnIRDialect>();
    context.getOrLoadDialect<::mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<::mlir::arith::ArithDialect>();
    context.getOrLoadDialect<::mlir::func::FuncDialect>();

    voidType = ::mlir::KUNAI::MjolnIR::DVMVoidType::get(&context);
    byteType = ::mlir::IntegerType::get(&context, 8);
    boolType = ::mlir::IntegerType::get(&context, 1);
    charType = ::mlir::IntegerType::get(&context, 16);
    shortType = ::mlir::IntegerType::get(&context, 16);
    intType = ::mlir::IntegerType::get(&context, 32);
    longType = ::mlir::IntegerType::get(&context, 64);
    floatType = ::mlir::Float32Type::get(&context);
    doubleType = ::mlir::Float64Type::get(&context);
    strObjectType = ::mlir::KUNAI::MjolnIR::DVMObjectType::get(&context, "Ljava/lang/String;");
}

mlir::Type Lifter::get_type(KUNAI::DEX::DVMFundamental *fundamental)
{
    switch (fundamental->get_fundamental_type())
    {
    case KUNAI::DEX::DVMFundamental::BOOLEAN:
        return boolType;
    case KUNAI::DEX::DVMFundamental::BYTE:
        return byteType;
    case KUNAI::DEX::DVMFundamental::CHAR:
        return charType;
    case KUNAI::DEX::DVMFundamental::DOUBLE:
        return doubleType;
    case KUNAI::DEX::DVMFundamental::FLOAT:
        return floatType;
    case KUNAI::DEX::DVMFundamental::INT:
        return intType;
    case KUNAI::DEX::DVMFundamental::LONG:
        return longType;
    case KUNAI::DEX::DVMFundamental::SHORT:
        return shortType;
    default:
        return voidType;
    }
}

mlir::Type Lifter::get_array(KUNAI::DEX::DVMType *type)
{
    return ::mlir::KUNAI::MjolnIR::DVMArrayType::get(&context, get_type(type));
}

mlir::Type Lifter::get_type(KUNAI::DEX::DVMArray *array)
{
    return ::mlir::KUNAI::MjolnIR::DVMArrayType::get(&context,
                                                     get_type(const_cast<KUNAI::DEX::DVMType *>(array->get_array_type())));
}

mlir::Type Lifter::get_type(KUNAI::DEX::DVMClass *cls)
{
    return ::mlir::KUNAI::MjolnIR::DVMObjectType::get(&context, cls->get_name());
}

mlir::Type Lifter::get_type(KUNAI::DEX::DVMType *type)
{
    if (type->get_type() == KUNAI::DEX::DVMType::FUNDAMENTAL)
        return get_type(reinterpret_cast<KUNAI::DEX::DVMFundamental *>(type));
    else if (type->get_type() == KUNAI::DEX::DVMType::CLASS)
        return get_type(reinterpret_cast<KUNAI::DEX::DVMClass *>(type));
    else if (type->get_type() == KUNAI::DEX::DVMType::ARRAY)
        return get_type(reinterpret_cast<KUNAI::DEX::DVMArray *>(type));
    else
        throw exceptions::LifterException("MjolnIRLifter::get_type: that type is unknown or I don't know what it is...");
}

llvm::SmallVector<mlir::Type> Lifter::gen_prototype(KUNAI::DEX::ProtoID *proto)
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

::mlir::func::FuncOp Lifter::get_method(KUNAI::DEX::MethodAnalysis *M)
{
    auto encoded_method = std::get<KUNAI::DEX::EncodedMethod *>(M->get_encoded_method());

    auto method = encoded_method->getMethodID();

    KUNAI::DEX::ProtoID *proto = method->get_proto();
    std::string &name = method->get_name();

    auto method_location = mlir::FileLineColLoc::get(&context, llvm::StringRef(name.c_str()), 0, 0);

    // now let's create a MethodOp, for that we will need first to retrieve
    // the type of the parameters
    auto paramTypes = gen_prototype(proto);

    mlir::FunctionType methodType;

    // now retrieve the return type
    // we have to check if the method returns void
    if (proto->get_return_type()->get_type() == KUNAI::DEX::DVMType::FUNDAMENTAL &&
        reinterpret_cast<KUNAI::DEX::DVMFundamental *>(proto->get_return_type())->get_fundamental_type() ==
            KUNAI::DEX::DVMFundamental::VOID)
    {
        methodType = builder.getFunctionType(paramTypes, mlir::TypeRange());
    }
    else
    {
        mlir::Type retType = get_type(proto->get_return_type());
        // create now the method type
        methodType = builder.getFunctionType(paramTypes, {retType});
    }

    auto methodOp = builder.create<::mlir::func::FuncOp>(method_location, name, methodType);

    auto entryBB = &methodOp.getBody().emplaceBlock();

    /// declare the register parameters, these are used during the
    /// program
    auto number_of_params = proto->get_parameters().size();

    auto number_of_registers = encoded_method->get_code_item().get_registers_size();

    auto first_block = M->get_basic_blocks().get_basic_block_by_idx(0);

    for (std::uint32_t Reg = (number_of_registers - number_of_params), /// starting index of the parameter
         Limit = (static_cast<std::uint32_t>(number_of_registers)),    /// limit value for parameters
         Argument = 0;                                                 /// for obtaining parameter by index 0
         Reg < Limit;
         ++Reg,
                       ++Argument)
    {
        /// generate and get the value from the parameter
        auto value = entryBB->addArgument(paramTypes[Argument], method_location);
        // auto value = methodOp.getArgument(Argument);
        /// write to a local variable
        writeLocalVariable(first_block, Reg, value);
    }

    // with the type created, now create the Method
    return methodOp;
}

mlir::Value Lifter::readLocalVariableRecursive(KUNAI::DEX::DVMBasicBlock *BB,
                                               KUNAI::DEX::BasicBlocks &BBs,
                                               std::uint32_t Reg)
{
    /// for checking the predecessor block
    auto _check_pred = [this](KUNAI::DEX::DVMBasicBlock *pred)
    {
        /// If the predecessor block was analyzed, it's okay
        if (CurrentDef[pred].Analyzed)
            return;
        /// in other case, generate it
        analysis_context.ip = builder.saveInsertionPoint();
        scope_context.push(analysis_context);
        gen_block(pred);
        analysis_context = scope_context.top();
        scope_context.pop();
        /// set again the insertion point
        builder.restoreInsertionPoint(analysis_context.ip);
    };

    mlir::Value new_value;

    /// because block doesn't have it add it to required.
    CurrentDef[BB].required.insert(Reg);

    auto predecessors = BBs.get_predecessors()[BB];

    if (predecessors.size() == 1)
        return readLocalVariable(*predecessors.begin(), BBs, Reg);

    for (auto pred : predecessors)
    {
        _check_pred(pred);

        auto Val = readLocalVariable(pred, BBs, Reg);

        if (!Val)
            continue;

        /// if the value is required, add the argument to the block
        /// write the local variable and erase from required
        if (CurrentDef[BB].required.find(Reg) != CurrentDef[BB].required.end())
        {
            auto Loc = mlir::FileLineColLoc::get(&context, module_name, BB->get_first_address(), 0);

            new_value = map_blocks[BB]->addArgument(Val.getType(), Loc);

            writeLocalVariable(BB, Reg, new_value);

            CurrentDef[BB].required.erase(Reg);
        }

        auto edge = std::make_pair(pred, BB);

        if (std::find(CurrentDef[pred].jmpParameters[edge].begin(),
                      CurrentDef[pred].jmpParameters[edge].end(),
                      Val) == CurrentDef[pred].jmpParameters[edge].end())
            CurrentDef[pred].jmpParameters[edge].push_back(Val);
    }

    return new_value;
}

// fills phi nodes with correct values, assumes block is sealed
void Lifter::fillBlockArgs(KUNAI::DEX::BasicBlocks &BBs, KUNAI::DEX::DVMBasicBlock *block)
{
    for (auto pred : BBs.predecessors(block))
    {
        if (pred->is_start_block() || pred->is_end_block())
            continue;

        assert(CurrentDef.find(pred) != CurrentDef.end() && "Checked block not found in CurrentDef");

        auto edge = std::make_pair(pred, block);

        ::set_block_args_for_predecessor(map_blocks[block], map_blocks[pred], CurrentDef[pred].jmpParameters[edge]);
    }
}

void Lifter::gen_method(KUNAI::DEX::MethodAnalysis *method)
{
    /// fix for whenever we have a fallthrough
    auto _gen_fallthrough_jmp = [this](KUNAI::DEX::DVMBasicBlock *bb,
                                       KUNAI::DEX::BasicBlocks::connected_blocks_t &sucessors,
                                       KUNAI::DEX::BasicBlocks &bbs)
    {
        /// if the block doesn't finish with a terminator instruction
        if (!bb->get_instructions().back()->is_terminator() &&
            /// and also it cannot be a right predecessor from an end block
            !(*sucessors[bb].begin())->is_end_block())
        {
            auto last_instr = bb->get_instructions().back();

            auto next_block = bbs.get_basic_block_by_idx(
                last_instr->get_address() + last_instr->get_instruction_length());

            auto loc = mlir::FileLineColLoc::get(&context, module_name, last_instr->get_address(), 1);

            builder.setInsertionPointToEnd(map_blocks[bb]);

            builder.create<mlir::cf::BranchOp>(
                loc,
                map_blocks[next_block]);
        }
    };

    /// create the method
    auto function = get_method(method);
    /// obtain the basic blocks
    auto &bbs = method->get_basic_blocks();
    /// update the current method
    analysis_context.current_method = method;

    /// generate the blocks for each node
    for (auto bb : bbs.get_nodes())
    {
        if (bb->is_start_block() || bb->is_end_block())
            continue;
        if (bb->get_first_address() == 0) // if it's the first block
        {
            auto &entryBlock = function.front();
            map_blocks[bb] = &entryBlock;
        }
        else // others must be generated
            map_blocks[bb] = function.addBlock();
    }
    /// now traverse each node for generating instructions
    for (auto bb : bbs.get_nodes())
    {
        /// if it's a starting block
        /// or it's an ending block
        /// or it's has already been analyzed
        if (bb->is_start_block() || bb->is_end_block() || CurrentDef[bb].Analyzed)
            continue;

        gen_block(bb);
    }

    auto &sucessors = bbs.get_sucessors();

    for (auto bb : bbs.get_nodes())
    {
        if (bb->is_start_block() || bb->is_end_block())
            continue;

        _gen_fallthrough_jmp(bb, sucessors, bbs);

        fillBlockArgs(bbs, bb);
    }
}

void Lifter::cast_to_type(std::uint32_t reg,
                          mlir::Type type,
                          mlir::FileLineColLoc loc)
{
    bool changed_value = false;
    mlir::Value new_value;
    auto curr_value = readLocalVariable(analysis_context.current_basic_block,
                                        analysis_context.current_method->get_basic_blocks(),
                                        reg);
    auto curr_type = curr_value.getType();

    if (type == curr_type) // if type is equal return
        return;

    if (::is_float(type)) /// type is float
    {
        if (::is_integer(curr_type)) /// int to float
        {
            new_value = builder.create<::mlir::arith::SIToFPOp>(
                loc,
                type,
                curr_value);
            changed_value = true;
        }

        if (::is_float(curr_type)) /// both are float
        {
            if (type.getIntOrFloatBitWidth() >
                curr_type.getIntOrFloatBitWidth()) /// extend float
            {
                new_value = builder.create<::mlir::arith::ExtFOp>(
                    loc,
                    type,
                    curr_value);
                changed_value = true;
            }
            else /// trunc float
            {
                new_value = builder.create<::mlir::arith::TruncFOp>(
                    loc,
                    type,
                    curr_value);
                changed_value = true;
            }
        }
    }
    else if (::is_integer(type))
    {
        if (::is_integer(curr_type))
        {
            if (type.getIntOrFloatBitWidth() >
                curr_type.getIntOrFloatBitWidth()) /// extend int
            {
                new_value = builder.create<::mlir::arith::ExtSIOp>(
                    loc,
                    type,
                    curr_value);
                changed_value = true;
            }
            else /// trunc integer
            {
                new_value = builder.create<::mlir::arith::TruncIOp>(
                    loc,
                    type,
                    curr_value);
                changed_value = true;
            }
        }

        if (::is_float(curr_type)) /// float to int
        {
            new_value = builder.create<::mlir::arith::FPToSIOp>(
                loc,
                type,
                curr_value);
            changed_value = true;
        }
    }

    if (changed_value)
    {
        /// write the new value as the current one
        writeLocalVariable(analysis_context.current_basic_block, reg, new_value);
    }
}

mlir::Value Lifter::cast_value(mlir::Value curr_value,
                               mlir::Type type,
                               mlir::FileLineColLoc loc)
{
    bool changed_value = false;
    mlir::Value new_value;
    auto curr_type = curr_value.getType();

    if (::is_float(type)) /// type is float
    {
        if (::is_integer(curr_type)) /// int to float
        {
            new_value = builder.create<::mlir::arith::SIToFPOp>(
                loc,
                type,
                curr_value);
            changed_value = true;
        }

        if (::is_float(curr_type)) /// both are float
        {
            if (type.getIntOrFloatBitWidth() >
                curr_type.getIntOrFloatBitWidth()) /// extend float
            {
                new_value = builder.create<::mlir::arith::ExtFOp>(
                    loc,
                    type,
                    curr_value);
                changed_value = true;
            }
            else /// trunc float
            {
                new_value = builder.create<::mlir::arith::TruncFOp>(
                    loc,
                    type,
                    curr_value);
                changed_value = true;
            }
        }
    }
    else if (::is_integer(type))
    {
        if (::is_integer(curr_type))
        {
            if (type.getIntOrFloatBitWidth() >
                curr_type.getIntOrFloatBitWidth()) /// extend int
            {
                new_value = builder.create<::mlir::arith::ExtSIOp>(
                    loc,
                    type,
                    curr_value);
            }
            else /// trunc integer
            {
                new_value = builder.create<::mlir::arith::TruncIOp>(
                    loc,
                    type,
                    curr_value);
            }
        }

        if (::is_float(curr_type)) /// float to int
        {
            new_value = builder.create<::mlir::arith::FPToSIOp>(
                loc,
                type,
                curr_value);
            changed_value = true;
        }
    }

    if (changed_value)
        return new_value;
    return curr_value;
}

void Lifter::cast_to_type(std::uint32_t reg1,
                          std::uint32_t reg2,
                          mlir::FileLineColLoc loc)
{
    bool changed_value1 = false;
    bool changed_value2 = false;

    mlir::Value new_value;

    auto curr_value1 = readLocalVariable(analysis_context.current_basic_block,
                                         analysis_context.current_method->get_basic_blocks(),
                                         reg1);
    auto curr_value2 = readLocalVariable(analysis_context.current_basic_block,
                                         analysis_context.current_method->get_basic_blocks(),
                                         reg2);

    auto curr_type1 = curr_value1.getType();
    auto curr_type2 = curr_value2.getType();

    if (curr_type1 == curr_type2)
        return;

    if (::is_float(curr_type1) && ::is_float(curr_type2))
    {
        if (curr_type1.getIntOrFloatBitWidth() > curr_type2.getIntOrFloatBitWidth())
        {
            new_value = builder.create<::mlir::arith::ExtFOp>(
                loc,
                curr_type1,
                curr_value2);
            changed_value2 = true;
        }
        else
        {
            new_value = builder.create<::mlir::arith::ExtFOp>(
                loc,
                curr_type2,
                curr_value1);
            changed_value1 = true;
        }
    }

    else if (::is_integer(curr_type1) && ::is_integer(curr_type2))
    {
        if (curr_type1.getIntOrFloatBitWidth() > curr_type2.getIntOrFloatBitWidth())
        {
            new_value = builder.create<::mlir::arith::ExtSIOp>(
                loc,
                curr_type1,
                curr_value2);
            changed_value2 = true;
        }
        else
        {
            new_value = builder.create<::mlir::arith::ExtSIOp>(
                loc,
                curr_type2,
                curr_value1);
            changed_value1 = true;
        }
    }

    else if (::is_float(curr_type1))
    {
        new_value = builder.create<::mlir::arith::SIToFPOp>(
            loc,
            curr_type1,
            curr_value2);
        changed_value2 = true;
    }

    else if (::is_float(curr_type2))
    {
        new_value = builder.create<::mlir::arith::SIToFPOp>(
            loc,
            curr_type2,
            curr_value1);
        changed_value1 = true;
    }

    if (changed_value1)
    {
        /// write the new value as the current one
        writeLocalVariable(analysis_context.current_basic_block, reg1, new_value);
    }
    else if (changed_value2)
    {
        /// write the new value as the current one
        writeLocalVariable(analysis_context.current_basic_block, reg2, new_value);
    }
}

void Lifter::gen_block(KUNAI::DEX::DVMBasicBlock *bb)
{
    /// set as the insertion point of the instructions
    builder.setInsertionPointToStart(map_blocks[bb]);

    /// update current basic blocks
    analysis_context.current_basic_block = bb;

    if (bb->is_try_block())
    {
        auto Loc = mlir::FileLineColLoc::get(&context, module_name, bb->get_first_address(), 0);
        builder.create<::mlir::KUNAI::MjolnIR::TryOp>(Loc);
    }
    else if (bb->is_catch_block())
    {
        auto Loc = mlir::FileLineColLoc::get(&context, module_name, bb->get_first_address(), 0);
        builder.create<::mlir::KUNAI::MjolnIR::CatchOp>(Loc);
    }

    for (auto instr : bb->get_instructions())
    {
        try
        {
            /// generate the instruction
            gen_instruction(instr);
        }
        catch (const exceptions::LifterException &e)
        {
            /// if user wants to generate exception
            if (gen_exception)
                throw e;
            /// if not just create a Nop instruction
            auto Loc = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);
            builder.create<::mlir::KUNAI::MjolnIR::Nop>(Loc);
        }
    }

    CurrentDef[bb].Analyzed = 1;
}

mlir::OwningOpRef<mlir::ModuleOp> Lifter::mlirGen(KUNAI::DEX::MethodAnalysis *methodAnalysis)
{
    /// create a Module
    Module = mlir::ModuleOp::create(builder.getUnknownLoc());

    /// Set the insertion point into the region of the Module
    builder.setInsertionPointToEnd(Module.getBody());

    if (module_name.empty())
        module_name = methodAnalysis->get_class_name();

    Module.setName(module_name);

    gen_method(methodAnalysis);

    return Module;
}