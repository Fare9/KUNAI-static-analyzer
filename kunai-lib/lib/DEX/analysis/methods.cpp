//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file methods.cpp
// @brief Definitions of methods from analysis.hpp

#include "Kunai/DEX/analysis/analysis.hpp"
#include "Kunai/Utils/logger.hpp"
#include "Kunai/DEX/DVM/disassembler.hpp"
#include "Kunai/DEX/DVM/dalvik_opcodes.hpp"

#include <queue>

using namespace KUNAI::DEX;

bool MethodAnalysis::is_android_api() const
{
    if (!is_external)
        return false;
    
    auto class_name = this->get_class_name();

    for (const auto &known_api : known_apis)
    {
        if (class_name.find(known_api) == 0)
            return true;
    }

    return false;
}

const std::string &MethodAnalysis::get_name() const
{
    if (!name.empty())
        return name;

    if (is_external)
        name = std::get<ExternalMethod *>(method_encoded)->get_name_idx();
    else
        name = std::get<EncodedMethod *>(method_encoded)->getMethodID()->get_name();

    return name;
}

const std::string &MethodAnalysis::get_descriptor() const
{
    if (!descriptor.empty())
        return descriptor;
    
    if (is_external)
        descriptor = std::get<ExternalMethod*>(method_encoded)->get_proto_idx();
    else
        descriptor = std::get<EncodedMethod*>(method_encoded)->getMethodID()->get_proto()->get_shorty_idx();
    
    return descriptor;
}

const std::string &MethodAnalysis::get_access_flags() const
{
    if (!access_flag.empty())
        return access_flag;
    
    if (is_external)
        access_flag = DalvikOpcodes::get_access_flags_str(std::get<ExternalMethod*>(method_encoded)->get_access_flags());
    else
        access_flag = DalvikOpcodes::get_method_access_flags(std::get<EncodedMethod*>(method_encoded));
    
    return access_flag;
}

const std::string &MethodAnalysis::get_class_name() const
{
    if (!class_name.empty())
        return class_name;
    
    if (is_external)
        class_name = std::get<ExternalMethod*>(method_encoded)->get_class_idx();
    else
        class_name = std::get<EncodedMethod*>(method_encoded)->getMethodID()->get_class()->get_raw();
    
    return class_name;
}

const std::string &MethodAnalysis::get_full_name() const
{
    if (!full_name.empty())
        return full_name;
    
    if (is_external)
        full_name = std::get<ExternalMethod*>(method_encoded)->pretty_method_name();
    else
        full_name = std::get<EncodedMethod*>(method_encoded)->getMethodID()->pretty_method();
        
    return full_name;
}

void MethodAnalysis::create_basic_blocks()
{
    /// utilities to create the basic blocks
    std::vector<std::int64_t> entry_points;
    std::unordered_map<std::uint64_t,
                       std::vector<std::int64_t>>
        targets_jumps;
    Disassembler disassembler;

    // some useful variables
    auto logger = LOGGER::logger();
    auto method = std::get<EncodedMethod *>(method_encoded);
    

    logger->debug("create_basic_blocks: started creating the basic blocks for method {}", 
        method->getMethodID()->pretty_method());

    // we always have an start block
    DVMBasicBlock *start = new DVMBasicBlock();
    
    start->set_start_block(true);
    basic_blocks.add_node(start);

    // create the first block
    DVMBasicBlock *current = new DVMBasicBlock();
    basic_blocks.add_edge(start, current);

    // detect the targets of the jumps and switches
    for (const auto &instruction : instructions)
    {
        auto operation = DalvikOpcodes::get_instruction_operation(instruction->get_instruction_opcode());

        if (operation == TYPES::Operation::CONDITIONAL_BRANCH_DVM_OPCODE ||
            operation == TYPES::Operation::UNCONDITIONAL_BRANCH_DVM_OPCODE ||
            operation == TYPES::Operation::MULTI_BRANCH_DVM_OPCODE)
        {
            auto idx = instruction->get_address();
            auto ins = instruction.get();

            auto v = disassembler.determine_next(ins, idx);
            targets_jumps[idx] = std::move(v);
            entry_points.insert(entry_points.begin(), targets_jumps[idx].begin(), targets_jumps[idx].end());
        }
    }

    // now analyze the exceptions and obtain the entry point addresses
    exceptions = disassembler.determine_exception(method);

    for (const auto & except : exceptions)
    {
        entry_points.push_back(except.try_value_start_addr);
        for (const auto & handler : except.handler)
        {
            entry_points.push_back(handler.handler_start_addr);
        }
    }

    for (const auto & instruction : instructions)
    {
        auto idx = instruction->get_address();
        auto ins = instruction.get();

        /// if we find a new entry point, create a new basic block
        if (std::find(entry_points.begin(), entry_points.end(), static_cast<std::int64_t>(idx)) != entry_points.end() &&
            current->get_nb_instructions() != 0)
        {
            current = new DVMBasicBlock();
            basic_blocks.add_node(current);
        }

        current->add_instruction(ins);
    }

    if (current->get_nb_instructions() == 0)
        basic_blocks.remove_node(current);
    
    /// add the jump targets
    for (const auto & jump_target : targets_jumps)
    {
        auto src_idx = jump_target.first;
        auto src = basic_blocks.get_basic_block_by_idx(src_idx);

        for (auto dst_idx : jump_target.second)
        {
            auto dst = basic_blocks.get_basic_block_by_idx(dst_idx);

            basic_blocks.add_edge(src, dst);
        }
    }

    /// now set the exceptions
    for (const auto & except : exceptions)
    {
        auto try_bb = basic_blocks.get_basic_block_by_idx(except.try_value_start_addr);
        try_bb->set_try_block(true);

        for (const auto & handler : except.handler)
        {
            auto catch_bb = basic_blocks.get_basic_block_by_idx(handler.handler_start_addr);
            catch_bb->set_catch_block(true);
        }
    }

}