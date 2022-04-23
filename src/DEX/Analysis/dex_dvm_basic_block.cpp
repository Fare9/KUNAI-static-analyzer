#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * DVMBasicBlock
         */
        DVMBasicBlock::DVMBasicBlock(std::uint64_t start,
                                     dalvikopcodes_t dalvik_opcodes,
                                     basicblocks_t context,
                                     encodedmethod_t method,
                                     std::map<std::uint64_t, instruction_t> &instructions) : start(start),
                                                                                                            end(start),
                                                                                                            dalvik_opcodes(dalvik_opcodes),
                                                                                                            context(context),
                                                                                                            method(method),
                                                                                                            instructions(instructions)
        {
            // get name for basic block
            this->name = *this->method->get_method()->get_method_name();

            std::stringstream stream;
            stream << std::hex << this->start;

            this->name += "-BB@" + stream.str();
        }

        std::vector<instruction_t> DVMBasicBlock::get_instructions()
        {
            std::vector<instruction_t> bb_instructions;

            for (auto instruction : instructions)
            {
                if ((start <= instruction.first) && (instruction.first < end))
                    bb_instructions.push_back(instruction.second);
            }

            return bb_instructions;
        }

        instruction_t DVMBasicBlock::get_last()
        {
            auto bb = get_instructions();
            return bb[bb.size() - 1];
        }

        void DVMBasicBlock::set_parent(std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock *> bb)
        {
            parents.push_back(bb);
        }

        void DVMBasicBlock::set_child()
        {
            auto next_block = context->get_basic_block_by_idx(end + 1);
            if (next_block != nullptr)
            {
                childs.push_back({end - last_length, end, next_block});
            }

            for (auto child : childs)
            {
                auto child_block = std::get<2>(child);
                if (child_block != nullptr)
                {
                    auto last_idx = std::get<1>(child);
                    auto end_idx = std::get<0>(child);
                    child_block->set_parent({last_idx, end_idx, this});
                }
            }
        }

        void DVMBasicBlock::set_child(const std::vector<int64_t> &values)
        {
            for (auto value : values)
            {
                if (value == -1)
                    continue;

                if (const auto next_block = context->get_basic_block_by_idx(value))
                    childs.push_back({end - last_length, value, next_block});
            }

            for (auto child : childs)
            {
                if (auto child_block = std::get<2>(child))
                {
                    auto last_idx = std::get<1>(child);
                    auto end_idx = std::get<0>(child);
                    child_block->set_parent({last_idx, end_idx, this});
                }
            }
        }

        void DVMBasicBlock::push(instruction_t instr)
        {
            nb_instructions += 1;
            std::uint64_t idx = end;
            last_length = instr->get_length();
            end += last_length;

            std::uint32_t op_value = instr->get_OP();

            if ((op_value == DVMTypes::Opcode::OP_FILL_ARRAY_DATA) ||
                (op_value == DVMTypes::Opcode::OP_PACKED_SWITCH) ||
                (op_value == DVMTypes::Opcode::OP_SPARSE_SWITCH))
            {
                auto i = reinterpret_cast<Instruction31t *>(instr.get());
                special_instructions[idx] = instructions[idx + i->get_offset() * 2];
            }
        }

        instruction_t DVMBasicBlock::get_special_instruction(std::uint64_t idx)
        {
            if (special_instructions.find(idx) != special_instructions.end())
                return special_instructions[idx];
            else
                return nullptr;
        }

        /**
         * BasicBlocks
         */
        BasicBlocks::BasicBlocks() {}

        BasicBlocks::~BasicBlocks()
        {
            if (!basic_blocks.empty())
                basic_blocks.clear();
        }

        void BasicBlocks::push_basic_block(dvmbasicblock_t basic_block)
        {
            this->basic_blocks.push_back(basic_block);
        }

        dvmbasicblock_t BasicBlocks::pop_basic_block()
        {
            dvmbasicblock_t last_bb = nullptr;
            if (this->basic_blocks.size() >= 1)
            {
                last_bb = this->basic_blocks[this->basic_blocks.size() - 1];
                this->basic_blocks.pop_back();
            }

            return last_bb;
        }

        dvmbasicblock_t BasicBlocks::get_basic_block_by_idx(std::uint64_t idx)
        {
            for (auto basic_block : basic_blocks)
            {
                if ((basic_block.get()->get_start() <= idx) && (idx < basic_block.get()->get_end()))
                {
                    return basic_block;
                }
            }

            return nullptr;
        }

    }
}