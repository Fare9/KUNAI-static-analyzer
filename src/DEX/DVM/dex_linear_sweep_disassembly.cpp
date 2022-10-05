#include "KUNAI/DEX/DVM/dex_linear_sweep_disassembly.hpp"

namespace KUNAI {
    namespace DEX {
        LinearSweepDisassembler::LinearSweepDisassembler(dalvikopcodes_t dalvik_opcodes) :
        dalvik_opcodes (dalvik_opcodes)
        {
        }

        std::map<std::uint64_t, instruction_t> LinearSweepDisassembler::disassembly(const std::vector<std::uint8_t>& byte_buffer)
        {
            auto logger = LOGGER::logger();
            std::map<std::uint64_t, instruction_t> instructions;
            std::uint64_t instruction_index = 0;
            instruction_t instruction;
            size_t buffer_size = byte_buffer.size();
            std::uint32_t opcode;
            bool exist_switch = false;

            std::stringstream input_buffer;

            input_buffer.write(reinterpret_cast<const char*>(byte_buffer.data()), buffer_size);

            while (instruction_index < buffer_size)
            {
                input_buffer.seekg(instruction_index, std::ios_base::beg);

                opcode = byte_buffer[instruction_index];

                try
                {
                    if (opcode == DVMTypes::OP_PACKED_SWITCH || opcode == DVMTypes::OP_SPARSE_SWITCH)
                        exist_switch = true;
                    instruction = get_instruction_object(opcode, this->dalvik_opcodes, input_buffer);

                    if (instruction)
                    {
                        instructions[instruction_index] = instruction;
                        instruction_index += instruction->get_length();
                    }
                }
                catch(const exceptions::InvalidInstruction& i)
                {
                    logger->error("InvalidInstruction in the index: {}, opcode: {}, message: {}, instruction size: {}", instruction_index, opcode, i.what(), i.size());
                    // Create a DalvikErrorInstruction
                    std::stringstream error_buffer;
                    error_buffer.write(reinterpret_cast<const char*>(byte_buffer.data() + instruction_index), i.size());
                    instruction = std::make_shared<DalvikIncorrectInstruction>(this->dalvik_opcodes, error_buffer, i.size());

                    // Set the instruction
                    instructions[instruction_index] = instruction;

                    // advance the index
                    instruction_index += i.size();
                }
                catch(const std::exception& e)
                {
                    logger->error("Error reading index: {}, opcode: {}, message: {}", instruction_index, opcode, e.what());
                    instruction_index += 1;
                }
            }

            if (exist_switch)
                assign_switch_if_any(instructions);

            return instructions;
        }

        void LinearSweepDisassembler::assign_switch_if_any(std::map<std::uint64_t, instruction_t>& instrs)
        {
            for (auto instr : instrs)
            {
                if (instr.second->get_OP() == DVMTypes::OP_PACKED_SWITCH || instr.second->get_OP() == DVMTypes::OP_SPARSE_SWITCH)
                {
                    auto instr31t = std::dynamic_pointer_cast<Instruction31t>(instr.second);

                    auto switch_idx = instr.first + (instr31t->get_offset()*2);

                    if (instrs.find(switch_idx) != instrs.end())
                    {
                        if (instr.second->get_OP() == DVMTypes::OP_PACKED_SWITCH)
                            instr31t->set_packed_switch(std::dynamic_pointer_cast<PackedSwitch>(instrs[switch_idx]));
                        else if (instr.second->get_OP() == DVMTypes::OP_SPARSE_SWITCH)
                            instr31t->set_sparse_switch(std::dynamic_pointer_cast<SparseSwitch>(instrs[switch_idx]));
                    }
                }
            }
        }
    }
}