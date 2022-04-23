#include "dex_linear_sweep_disassembly.hpp"

namespace KUNAI {
    namespace DEX {
        LinearSweepDisassembler::LinearSweepDisassembler(dalvikopcodes_t dalvik_opcodes) :
        dalvik_opcodes (dalvik_opcodes)
        {
        }

        std::map<std::uint64_t, instruction_t> LinearSweepDisassembler::disassembly(const std::vector<std::uint8_t>& byte_buffer)
        {
            std::map<std::uint64_t, instruction_t> instructions;
            std::uint64_t instruction_index = 0;
            instruction_t instruction;
            size_t buffer_size = byte_buffer.size();
            std::uint32_t opcode;
            bool exist_switch = false;

            while (instruction_index < buffer_size)
            {
                std::stringstream input_buffer;

                opcode = byte_buffer[instruction_index];
                
                input_buffer.write(reinterpret_cast<const char*>(byte_buffer.data() + instruction_index), buffer_size - instruction_index);

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
                catch(const std::exception& e)
                {
                    std::cerr << "Error reading index " << instruction_index << " opcode " << opcode << " message: '";
                    std::cerr << e.what() << "'\n";
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