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
        descriptor = std::get<ExternalMethod *>(method_encoded)->get_proto_idx();
    else
        descriptor = std::get<EncodedMethod *>(method_encoded)->getMethodID()->get_proto()->get_shorty_idx();

    return descriptor;
}

const std::string &MethodAnalysis::get_access_flags() const
{
    if (!access_flag.empty())
        return access_flag;

    if (is_external)
        access_flag = DalvikOpcodes::get_access_flags_str(std::get<ExternalMethod *>(method_encoded)->get_access_flags());
    else
        access_flag = DalvikOpcodes::get_method_access_flags(std::get<EncodedMethod *>(method_encoded));

    return access_flag;
}

const std::string &MethodAnalysis::get_class_name() const
{
    if (!class_name.empty())
        return class_name;

    if (is_external)
        class_name = std::get<ExternalMethod *>(method_encoded)->get_class_idx();
    else
        class_name = std::get<EncodedMethod *>(method_encoded)->getMethodID()->get_class()->get_raw();

    return class_name;
}

const std::string &MethodAnalysis::get_full_name() const
{
    if (!full_name.empty())
        return full_name;

    if (is_external)
        full_name = std::get<ExternalMethod *>(method_encoded)->pretty_method_name();
    else
        full_name = std::get<EncodedMethod *>(method_encoded)->getMethodID()->pretty_method();

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
    for (const auto &instruction : instructions_)
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
            entry_points.insert(entry_points.end(), targets_jumps[idx].begin(), targets_jumps[idx].end());
        }
    }

    // now analyze the exceptions and obtain the entry point addresses
    exceptions = disassembler.determine_exception(method);

    for (const auto &except : exceptions)
    {
        /// entry point of try values can start in the middle
        /// of a block
        /// entry_points.push_back(except.try_value_start_addr);
        for (const auto &handler : except.handler)
        {
            entry_points.push_back(handler.handler_start_addr);
        }
    }

    for (const auto &instruction : instructions_)
    {
        auto idx = instruction->get_address();
        auto ins = instruction.get();

        /// if we find a new entry point, create a new basic block
        if (std::find(entry_points.begin(), entry_points.end(), static_cast<std::int64_t>(idx)) != entry_points.end() &&
            current->get_nb_instructions() != 0)
        {
            auto prev = current;
            current = new DVMBasicBlock();
            /// if last instruction is not a terminator
            /// we must create an edge because it comes
            /// from a fallthrough block
            if (!prev->get_terminator())
                basic_blocks.add_edge(prev, current);
            /// in other case, just add the node, and later
            /// will be added the edge
            else
                basic_blocks.add_node(current);
        }

        current->add_instruction(ins);
    }

    // check it is not a start
    if (current->get_nb_instructions() == 0)
        basic_blocks.remove_node(current);

    auto out_range = instructions_.back()->get_address() +
                     instructions_.back()->get_instruction_length();

    /// add the jump targets
    for (const auto &jump_target : targets_jumps)
    {
        auto src_idx = jump_target.first;
        auto src = basic_blocks.get_basic_block_by_idx(src_idx);
        /// need to check how target jump is generated
        if (src_idx >= out_range || src == nullptr)
            continue;

        for (auto dst_idx : jump_target.second)
        {
            auto dst = basic_blocks.get_basic_block_by_idx(dst_idx);
            /// need to check how target jump is generated
            if (dst_idx >= out_range || dst == nullptr)
                continue;

            basic_blocks.add_edge(src, dst);
        }
    }

    /// now set the exceptions
    for (const auto &except : exceptions)
    {
        auto try_bb = basic_blocks.get_basic_block_by_idx(except.try_value_start_addr);
        try_bb->set_try_block(true);

        for (const auto &handler : except.handler)
        {
            auto catch_bb = basic_blocks.get_basic_block_by_idx(handler.handler_start_addr);
            catch_bb->set_catch_block(true);
            catch_bb->set_handler_type(handler.handler_type);
        }
    }

    // we always finish with an ending block
    DVMBasicBlock *end = new DVMBasicBlock();

    end->set_end_block(true);
    basic_blocks.add_node(end);

    for (auto &node : basic_blocks.get_nodes())
    {
        if (node->is_start_block() || node->is_end_block())
            continue;

        /// if the node has not predecessors, add start
        /// node as its predecessor
        if (basic_blocks.get_predecessors()[node].size() == 0)
            basic_blocks.add_edge(start, node);

        /// if the node has not sucessors, add end node
        /// as its sucessor
        if (basic_blocks.get_sucessors()[node].size() == 0)
            basic_blocks.add_edge(node, end);
    }
}

void MethodAnalysis::dump_instruction_dot(std::ofstream &dot_file, Instruction *instr)
{

    dot_file << "<tr><td align=\"left\">";

    dot_file << std::right << std::setfill('0') << std::setw(8) << std::hex << instr->get_address() << "  ";

    const auto &opcodes = instr->get_opcodes();

    if (opcodes.size() > 8)
    {
        auto remaining = 8 - (opcodes.size() % 8);

        size_t aux = 0;

        for (const auto opcode : opcodes)
        {
            dot_file << std::right << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)opcode << " ";
            aux++;
            if (aux % 8 == 0)
            {
                dot_file << "\\l"
                         << "          ";
            }
        }

        for (std::uint8_t i = 0; i < remaining; i++)
            dot_file << "   ";
    }
    else
    {
        for (const auto opcode : opcodes)
            dot_file << std::right << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)opcode << " ";

        for (std::uint8_t i = 0, remaining_size = 8 - opcodes.size(); i < remaining_size; ++i)
            dot_file << "   ";
    }

    auto content = instr->print_instruction();

    while (content.find("\"") != content.npos)
        content.replace(content.find("\""), 1, "'", 1);

    while (content.find("<") != content.npos)
        content.replace(content.find("<"), 1, "&lt;", 4);

    while (content.find(">") != content.npos)
        content.replace(content.find(">"), 1, "&gt;", 4);

    dot_file << content << "</td></tr>\n";
}

void MethodAnalysis::dump_block_dot(std::ofstream &dot_file, DVMBasicBlock *bb)
{
    dot_file << "\"" << bb->get_name() << "\""
             << "[label=<<table border=\"0\" cellborder=\"0\" cellspacing=\"0\">\n";

    dot_file << "<tr><td colspan=\"2\" align=\"left\"><b>[" << bb->get_name() << "]</b></td></tr>\n";
    if (bb->is_try_block())
        dot_file << "<tr><td colspan=\"2\" align=\"left\"><b>try_:</b></td></tr>\n";
    else if (bb->is_catch_block())
        dot_file << "<tr><td colspan=\"2\" align=\"left\"><b>catch_:</b></td></tr>\n";

    for (auto instr : bb->get_instructions())
        dump_instruction_dot(dot_file, instr);

    dot_file << "</table>>];\n\n";
}

void MethodAnalysis::dump_method_dot(std::ofstream &dot_file)
{
    Disassembler d;

    if (full_name.empty())
        get_full_name();

    // first dump the headers of the dot file
    dot_file << "digraph \"" << full_name << "\"{\n";
    dot_file << "style=\"dashed\";\n";
    dot_file << "color=\"black\";\n";
    dot_file << "label=\"" << full_name << "\";\n";
    dot_file << "node [shape=box, style=filled, fillcolor=lightgrey, fontname=\"Courier\", fontsize=\"10\"];\n";
    dot_file << "edge [color=black, arrowhead=open];\n";

    for (auto bb : basic_blocks.get_nodes())
        dump_block_dot(dot_file, bb);

    for (const auto &edge : basic_blocks.get_edges())
    {
        auto terminator_instr = edge.first->get_terminator();

        if (terminator_instr &&
            DalvikOpcodes::get_instruction_operation(terminator_instr->get_instruction_opcode()) ==
                TYPES::Operation::CONDITIONAL_BRANCH_DVM_OPCODE)
        {
            auto second_block_addr = edge.second->get_first_address();

            if (second_block_addr ==
                (terminator_instr->get_address() + terminator_instr->get_instruction_length())) // falthrough branch
                dot_file << "\"" << edge.first->get_name() << "\" -> "
                         << "\"" << edge.second->get_name() << "\" [style=\"solid,bold\",color=red,weight=10,constraint=true];\n";
            else
                dot_file << "\"" << edge.first->get_name() << "\" -> "
                         << "\"" << edge.second->get_name() << "\" [style=\"solid,bold\",color=green,weight=10,constraint=true];\n";
        }
        else if (terminator_instr &&
                 DalvikOpcodes::get_instruction_operation(terminator_instr->get_instruction_opcode()) ==
                     TYPES::Operation::UNCONDITIONAL_BRANCH_DVM_OPCODE)
        {
            dot_file << "\"" << edge.first->get_name() << "\" -> "
                     << "\"" << edge.second->get_name() << "\" [style=\"solid,bold\",color=blue,weight=10,constraint=true];\n";
        }
        else
            dot_file << "\"" << edge.first->get_name() << "\" -> "
                     << "\"" << edge.second->get_name() << "\" [style=\"solid,bold\",color=black,weight=10,constraint=true];\n";
    }

    dot_file << "}";
}