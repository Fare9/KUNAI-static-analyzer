#include "KUNAI/DEX/DVM/dex_recursive_traversal_disassembly.hpp"

namespace KUNAI
{
    namespace DEX
    {
        RecursiveTraversalDisassembler::RecursiveTraversalDisassembler(DalvikOpcodes *dalvik_opcodes) : dalvik_opcodes(dalvik_opcodes)
        {
        }

        std::map<std::uint64_t, instruction_t> RecursiveTraversalDisassembler::disassembly(const std::vector<std::uint8_t> &byte_buffer, EncodedMethod *method)
        {
            auto logger = LOGGER::logger();
            std::map<std::uint64_t, instruction_t> instructions;
            std::map<std::uint64_t, bool> seen;
            std::uint64_t instruction_index;
            instruction_t instruction;
            size_t buffer_size = byte_buffer.size();
            std::uint32_t opcode;
            bool exist_switch = false;
            std::stringstream input_buffer;

            auto exceptions = determine_exception(dalvik_opcodes, method);

            input_buffer.write(reinterpret_cast<const char *>(byte_buffer.data()), buffer_size);

            // take every method start at index 0
            Q.push(0);

            // now all the handlers from the exceptions!
            for (auto &exception : exceptions)
            {
                // add the try parts
                Q.push(exception.try_value_start_addr);

                // add now the catches
                for (auto &handler : exception.handler)
                    Q.push(handler.handler_start_addr);
            }

#ifdef DEBUG
            logger->debug("Starting recursive traversal");
#endif

            while (!Q.empty())
            {
                // obtain the first address from the queue
                instruction_index = Q.front();
                Q.pop();

#ifdef DEBUG
                logger->debug("Disassembly from offset {}", instruction_index);
#endif
                // if we have already seen that address
                if (seen[instruction_index])
                    continue;

                while (instruction_index < buffer_size)
                {
                    try
                    {
                        input_buffer.seekg(instruction_index, std::ios_base::beg);

                        opcode = byte_buffer[instruction_index];

                        instruction = get_instruction_object(opcode, this->dalvik_opcodes, input_buffer);
                        instructions[instruction_index] = std::move(instruction);
                        seen[instruction_index] = true;

                        auto current_instr = instructions[instruction_index].get();

                        auto operation = dalvik_opcodes->get_instruction_operation(current_instr->get_OP());
                        // conditional jump
                        if (operation == DVMTypes::Operation::CONDITIONAL_BRANCH_DVM_OPCODE ||
                            // unconditional jump
                            operation == DVMTypes::Operation::UNCONDITIONAL_BRANCH_DVM_OPCODE ||
                            // switch instructions
                            operation == DVMTypes::Operation::MULTI_BRANCH_DVM_OPCODE)
                        {
                            if (operation == DVMTypes::Operation::MULTI_BRANCH_DVM_OPCODE)
                                analyze_switch(instructions, byte_buffer, current_instr, instruction_index);

                            auto next_offsets = determine_next(current_instr, instruction_index);

                            for (auto next_offset : next_offsets)
                            {
                                Q.push(next_offset);
                            }

                            break;
                        }

                        instruction_index += current_instr->get_length();
                    }
                    catch (const exceptions::InvalidInstruction &i)
                    {
                        logger->error("InvalidInstruction in the index: {}, opcode: {}, message: {}, instruction size: {}", instruction_index, opcode, i.what(), i.size());
                        // Create a DalvikErrorInstruction
                        std::stringstream error_buffer;
                        error_buffer.write(reinterpret_cast<const char *>(byte_buffer.data() + instruction_index), i.size());
                        instruction = std::make_unique<DalvikIncorrectInstruction>(this->dalvik_opcodes, error_buffer, i.size());

                        // Set the instruction
                        instructions[instruction_index] = std::move(instruction);

                        // advance the index
                        instruction_index += i.size();
                    }
                    catch (const std::exception &e)
                    {
                        logger->error("Error reading index: {}, opcode: {}, message: {}", instruction_index, opcode, e.what());
                        instruction_index += 1;
                    }
                }
            }

            return instructions;
        }

        void RecursiveTraversalDisassembler::analyze_switch(std::map<std::uint64_t, instruction_t> &instrs, const std::vector<std::uint8_t> &byte_buffer, Instruction *instruction, std::uint64_t instruction_index)
        {
            // get the instruction
            auto instr31t = reinterpret_cast<Instruction31t *>(instruction);

            auto switch_idx = instruction_index + (instr31t->get_offset() * 2);

            std::stringstream input_buffer;

            auto opcode = byte_buffer[switch_idx];

            input_buffer.write(reinterpret_cast<const char *>(byte_buffer.data() + switch_idx), byte_buffer.size() - switch_idx);

            auto new_instruction = get_instruction_object(opcode, this->dalvik_opcodes, input_buffer);

            instrs[switch_idx] = std::move(new_instruction);

            if (instruction->get_OP() == DVMTypes::OP_PACKED_SWITCH)
                instr31t->set_packed_switch(reinterpret_cast<PackedSwitch *>(instrs[switch_idx].get()));
            else if (instruction->get_OP() == DVMTypes::OP_SPARSE_SWITCH)
                instr31t->set_sparse_switch(reinterpret_cast<SparseSwitch *>(instrs[switch_idx].get()));
        }

    }
}