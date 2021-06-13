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

            std::stringstream input_buffer;

            while (instruction_index < buffer_size)
            {
                opcode = byte_buffer[instruction_index];
                std::copy(byte_buffer.begin() + instruction_index, byte_buffer.end(), std::ostream_iterator<std::uint8_t>(input_buffer));

                instruction = get_instruction_object(opcode, this->dalvik_opcodes, input_buffer);

                if (instruction)
                {
                    instructions[instruction_index] = instruction;
                    instruction_index += instruction->get_length();
                }
            }

            return instructions;
        }
    }
}