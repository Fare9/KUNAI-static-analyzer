#include "dex_linear_sweep_disassembly.hpp"

namespace KUNAI {
    namespace DEX {
        LinearSweepDisassembler::LinearSweepDisassembler(std::shared_ptr<DalvikOpcodes> dalvik_opcodes)
        {
            this->dalvik_opcodes = dalvik_opcodes;
        }

        LinearSweepDisassembler::~LinearSweepDisassembler() {}

        std::map<std::uint64_t, std::shared_ptr<Instruction>> LinearSweepDisassembler::disassembly(std::vector<std::uint8_t> byte_buffer)
        {
            std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions;
            std::uint64_t instruction_index = 0;
            std::shared_ptr<Instruction> instruction;
            size_t buffer_size = byte_buffer.size();
            std::uint32_t opcode;

            while (instruction_index < buffer_size)
            {
                std::stringstream input_buffer;

                opcode = byte_buffer[instruction_index];
                
                input_buffer.write(reinterpret_cast<const char*>(byte_buffer.data() + instruction_index), buffer_size - instruction_index);

                try
                {
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

            return instructions;
        }
    }
}