//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRLifter.cpp
#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"

using namespace KUNAI::MjolnIR;

mlir::Type Lifter::get_type(KUNAI::DEX::DVMFundamental *fundamental)
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

mlir::Type Lifter::get_type(KUNAI::DEX::DVMClass *cls)
{
    return ::mlir::KUNAI::MjolnIR::DVMObjectType::get(&context, cls->get_name());
}

mlir::Type Lifter::get_type(KUNAI::DEX::DVMType *type)
{
    if (type->get_type() == KUNAI::DEX::DVMType::FUNDAMENTAL)
        return get_type(reinterpret_cast<KUNAI::DEX::DVMFundamental *>(type));
    else if (type->get_type() == KUNAI::DEX::DVMType::CLASS)
        return get_type(reinterpret_cast<KUNAI::DEX::DVMClass*>(type));
    else if (type->get_type() == KUNAI::DEX::DVMType::ARRAY)
        throw exceptions::LifterException("MjolnIRLIfter::get_type: type ARRAY not implemented yet...");
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

::mlir::KUNAI::MjolnIR::MethodOp Lifter::get_method(KUNAI::DEX::MethodAnalysis *M)
{
    auto encoded_method = std::get<KUNAI::DEX::EncodedMethod *>(M->get_encoded_method());

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

    auto first_block = M->get_basic_blocks().get_basic_block_by_idx(0);

    for (std::uint32_t Reg = (number_of_registers - number_of_params), /// starting index of the parameter
         Limit = (static_cast<std::uint32_t>(number_of_registers)),    /// limit value for parameters
         Argument = 0;                                                 /// for obtaining parameter by index 0
         Reg < Limit;
         ++Reg,
                       ++Argument)
    {
        /// get the value from the parameter
        auto value = methodOp.getArgument(Argument);
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
    mlir::Value new_value;

    /// because block doesn't have it add it to required.
    CurrentDef[BB].required.insert(Reg);

    for (auto pred : BBs.get_predecessors()[BB])
    {
        if (!CurrentDef[pred].Analyzed)
            gen_block(pred);
        auto Val = readLocalVariable(pred, BBs, Reg);

        /// if the value is required, add the argument to the block
        /// write the local variable and erase from required
        if (CurrentDef[BB].required.find(Reg) != CurrentDef[BB].required.end())
        {
            auto Loc = mlir::FileLineColLoc::get(&context, module_name, BB->get_first_address(), 0);

            new_value = map_blocks[BB]->addArgument(Val.getType(), Loc);

            writeLocalVariable(BB, Reg, new_value);

            CurrentDef[BB].required.erase(Reg);
        }

        CurrentDef[pred].jmpParameters[std::make_pair(pred, BB)].push_back(Val);
    }

    return new_value;
}

void Lifter::gen_method(KUNAI::DEX::MethodAnalysis *method)
{
    /// create the method
    auto function = get_method(method);
    /// obtain the basic blocks
    auto &bbs = method->get_basic_blocks();
    /// update the current method
    current_method = method;

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
        if (bb->is_start_block() || bb->is_end_block())
            continue;
        /// set as the insertion point of the instructions
        builder.setInsertionPointToStart(map_blocks[bb]);

        gen_block(bb);
    }

    for (auto bb : bbs.get_nodes())
    {
        if (bb->is_start_block() || bb->is_end_block())
            continue;

        gen_terminators(bb);
    }
}

void Lifter::gen_block(KUNAI::DEX::DVMBasicBlock *bb)
{
    /// update current basic block
    current_basic_block = bb;

    for (auto instr : bb->get_instructions())
    {
        try
        {
            auto operation = KUNAI::DEX::DalvikOpcodes::get_instruction_operation(instr->get_instruction_opcode());
            /// we will generate terminators later
            if (instr->is_terminator() &&
                operation != KUNAI::DEX::TYPES::Operation::RET_BRANCH_DVM_OPCODE)
                continue;
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

void Lifter::gen_terminators(KUNAI::DEX::DVMBasicBlock *bb)
{
    current_basic_block = bb;

    auto last_instr = bb->get_instructions().back();

    builder.setInsertionPointToEnd(map_blocks[bb]);
    try
    {
        auto operation = KUNAI::DEX::DalvikOpcodes::get_instruction_operation(last_instr->get_instruction_opcode());
        
        if (operation == KUNAI::DEX::TYPES::Operation::RET_BRANCH_DVM_OPCODE)
            return;
        if (last_instr->is_terminator())
            gen_instruction(last_instr);
        else
        {
            auto next_block = current_method->get_basic_blocks().get_basic_block_by_idx(
                last_instr->get_address() + last_instr->get_instruction_length());

            auto loc = mlir::FileLineColLoc::get(&context, module_name, last_instr->get_address(), 1);

            builder.create<::mlir::KUNAI::MjolnIR::FallthroughOp>(
                loc,
                map_blocks[next_block],
                CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, next_block)]);
        }
    }
    catch (const exceptions::LifterException &e)
    {
        /// if user wants to generate exception
        if (gen_exception)
            throw e;
        /// if not just create a Nop instruction
        auto Loc = mlir::FileLineColLoc::get(&context, module_name, last_instr->get_address(), 0);
        builder.create<::mlir::KUNAI::MjolnIR::Nop>(Loc);
    }
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