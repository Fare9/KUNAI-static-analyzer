//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file linear_sweep_disassembler.cpp

#include "Kunai/DEX/DVM/linear_sweep_disassembler.hpp"
#include "Kunai/Exceptions/disassembler_exception.hpp"
#include "Kunai/Exceptions/invalidinstruction_exception.hpp"
#include "Kunai/Utils/logger.hpp"

#include <unordered_map>

using namespace KUNAI::DEX;


namespace
{
    bool compare_by_address(const std::unique_ptr<Instruction>& a, 
                            const std::unique_ptr<Instruction>& b)
    {
        return a->get_address() < b->get_address();
    }
}


void LinearSweepDisassembler::disassembly(std::vector<std::uint8_t> &buffer_bytes,
                                          std::vector<std::unique_ptr<Instruction>> &instructions)
{
    auto logger = LOGGER::logger();
    std::unordered_map<std::uint64_t, Instruction *> cache_instr; // cache for searching for switch instructions
    std::uint64_t idx = 0;                         // index of the instr
    std::unique_ptr<Instruction> instr;            // insruction to create
    auto buffer_size = buffer_bytes.size();        // size of the buffer
    std::uint32_t opcode;                          // opcode of the operation
    bool exist_switch = false;                     // check a switch exist

    while (idx < buffer_size)
    {
        opcode = static_cast<std::uint32_t>(buffer_bytes[idx]);

        try
        {
            if (!exist_switch &&
                (opcode == TYPES::opcodes::OP_PACKED_SWITCH ||
                 opcode == TYPES::opcodes::OP_SPARSE_SWITCH))
                exist_switch = true;

            instr = disassembler->disassemble_instruction(
                opcode,
                buffer_bytes,
                idx);

            if (instr)
            {
                instr->set_address(idx);

                instructions.push_back(std::move(instr));

                cache_instr[idx] = instructions.back().get();

                idx += instructions.back()->get_instruction_length();
            }
        }
        catch (const exceptions::InvalidInstructionException &i)
        {
            logger->error("InvalidInstructionException in the index: {}, opcode: {}, message: {}, instr size: {}",
                          idx, opcode, i.what(), i.size());
            // in case there was an invalid instr
            // create a DalvikIncorrectInstruction
            instr = std::make_unique<DalvikIncorrectInstruction>(buffer_bytes, idx, i.size());

            instr->set_address(idx);

            // set the instr into the vector
            instructions.push_back(std::move(instr));

            cache_instr[idx] = instructions.back().get();

            idx += i.size();
        }
        catch (const std::exception &e)
        {
            logger->error("Error reading index: {}, opcode: {}, message: {}",
                          idx, opcode, e.what());
            idx += 1;
        }
    }

    if (exist_switch)
        assign_switch_if_any(instructions, cache_instr);
    
    std::sort(instructions.begin(), instructions.end(), ::compare_by_address);
}

void LinearSweepDisassembler::assign_switch_if_any(
    std::vector<std::unique_ptr<Instruction>> &instructions,
    std::unordered_map<std::uint64_t, Instruction *> &cache_instructions)
{
    for (auto & instr : instructions)
    {
        auto op_code = instr->get_instruction_opcode();
        
        if (op_code == TYPES::opcodes::OP_PACKED_SWITCH ||
            op_code == TYPES::opcodes::OP_SPARSE_SWITCH)
        {
            auto instr31t = reinterpret_cast<Instruction31t*>(instr.get());

            auto switch_idx = instr31t->get_address() + (instr31t->get_offset() * 2);

            auto it = cache_instructions.find(switch_idx);

            if (it != cache_instructions.end())
            {
                if (op_code == TYPES::opcodes::OP_PACKED_SWITCH)
                    instr31t->set_packed_switch(reinterpret_cast<PackedSwitch*>(it->second));
                else if (op_code == TYPES::opcodes::OP_SPARSE_SWITCH)
                    instr31t->set_sparse_switch(reinterpret_cast<SparseSwitch*>(it->second));
            }
        }
    }
}