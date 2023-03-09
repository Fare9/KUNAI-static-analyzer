//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file recursive_traversal_disassembler.cpp

#include "Kunai/DEX/DVM/recursive_traversal_disassembler.hpp"
#include "Kunai/Exceptions/disassembler_exception.hpp"
#include "Kunai/Exceptions/invalidinstruction_exception.hpp"
#include "Kunai/DEX/DVM/dalvik_opcodes.hpp"

using namespace KUNAI::DEX;

namespace
{
    bool compare_by_address(const std::unique_ptr<Instruction>& a, 
                            const std::unique_ptr<Instruction>& b)
    {
        return a->get_address() < b->get_address();
    }
}

void RecursiveTraversalDisassembler::disassembly(std::vector<std::uint8_t> &buffer_bytes,
                                                 EncodedMethod *method,
                                                 std::vector<std::unique_ptr<Instruction>> &instructions)
{
    auto logger = LOGGER::logger();
    // cache instructions for quick searching
    std::unordered_map<std::uint64_t, Instruction *> cache_instrs;
    // unordered map for keeping track of analyzed
    // we keep an index and a bollean value
    std::unordered_map<std::uint64_t, bool> seen;
    // index of the instruction
    std::uint64_t idx = 0;
    // instruction pointer
    std::unique_ptr<Instruction> instruction;
    // size of the buffer
    auto buffer_size = buffer_bytes.size();
    // opcode
    std::uint32_t opcode;
    // check if the method contains a switch
    bool exist_switch = false;

    auto exceptions = disassembler->determine_exception(method);

    // take every method start at index 0
    Q.push(0);

    // now all the handlers from the exceptions
    for (auto &exception : exceptions)
    {
        // add try parts
        Q.push(exception.try_value_start_addr);

        // add now the catches
        for (auto &handler : exception.handler)
            Q.push(handler.handler_start_addr);
    }

    while (!Q.empty())
    {
        // obtain always the first from the queue
        idx = Q.front();
        Q.pop();

        // in case it is a seen address
        // continue
        if (seen[idx])
            continue;

        while (idx < buffer_size)
        {

            try
            {
                /// classical linear sweep disassembly
                opcode = buffer_bytes[idx];

                instruction = disassembler->disassemble_instruction(opcode, buffer_bytes, idx);

                if (instruction)
                {
                    instruction->set_address(idx);

                    instructions.push_back(std::move(instruction));

                    cache_instrs[idx] = instructions.back().get();
                }

                seen[idx] = true;

                auto operation = DalvikOpcodes::get_instruction_operation(opcode);

                /// analyze in case of FILL_ARRAY_DATA, each one
                /// we have to disassembly the data
                if (opcode == TYPES::opcodes::OP_FILL_ARRAY_DATA)
                {
                    auto fill_array_data = reinterpret_cast<Instruction31t *>(instructions.back().get());

                    Q.push(idx + (fill_array_data->get_offset() * 2));
                }

                if (
                    // conditional jump
                    operation == TYPES::Operation::CONDITIONAL_BRANCH_DVM_OPCODE ||
                    // unconditional jump
                    operation == TYPES::Operation::UNCONDITIONAL_BRANCH_DVM_OPCODE ||
                    // switch instructions
                    operation == TYPES::Operation::MULTI_BRANCH_DVM_OPCODE)
                {
                    if (operation == TYPES::Operation::MULTI_BRANCH_DVM_OPCODE)
                        analyze_switch(instructions, cache_instrs, buffer_bytes);

                    auto next_offsets = disassembler->determine_next(cache_instrs[idx], idx);

                    for (auto next_offset : next_offsets)
                    {
                        if (!seen[next_offset])
                            Q.push(next_offset);
                    }

                    if (operation == TYPES::Operation::UNCONDITIONAL_BRANCH_DVM_OPCODE)
                        break;
                }

                idx += cache_instrs[idx]->get_instruction_length();
            }
            catch (const exceptions::InvalidInstructionException &i)
            {
                logger->error("InvalidInstructionException in the index: {}, opcode: {}, message: {}, instruction size: {}",
                              idx, opcode, i.what(), i.size());
                // in case there was an invalid instruction
                // create a DalvikIncorrectInstruction
                instruction = std::make_unique<DalvikIncorrectInstruction>(buffer_bytes, idx, i.size());

                // set the instruction into the vector
                instructions.push_back(std::move(instruction));

                cache_instrs[idx] = instructions.back().get();

                idx += i.size();
            }
            catch (const std::exception &e)
            {
                logger->error("Error reading index: {}, opcode: {}, message: {}", idx, opcode, e.what());
                idx += 1;
            }
        }
    }

    std::sort(instructions.begin(), instructions.end(), ::compare_by_address);
}

void RecursiveTraversalDisassembler::analyze_switch(
    std::vector<std::unique_ptr<Instruction>> &instructions,
    std::unordered_map<std::uint64_t, Instruction *> &cache_instrs,
    std::vector<std::uint8_t> &buffer_bytes)
{
    auto instr31t = reinterpret_cast<Instruction31t *>(instructions.back().get());

    auto switch_idx = instr31t->get_address() + (instr31t->get_offset() * 2);

    auto opcode = buffer_bytes[switch_idx];

    auto new_instruction = disassembler->disassemble_instruction(opcode, buffer_bytes, switch_idx);

    new_instruction->set_address(switch_idx);

    instructions.push_back(std::move(new_instruction));

    cache_instrs[switch_idx] = instructions.back().get();

    if (instr31t->get_type_of_switch() == Instruction31t::type_of_switch_t::PACKED_SWITCH)
        instr31t->set_packed_switch(reinterpret_cast<PackedSwitch *>(cache_instrs[switch_idx]));
    else if (instr31t->get_type_of_switch() == Instruction31t::type_of_switch_t::SPARSE_SWITCH)
        instr31t->set_sparse_switch(reinterpret_cast<SparseSwitch *>(cache_instrs[switch_idx]));
}
