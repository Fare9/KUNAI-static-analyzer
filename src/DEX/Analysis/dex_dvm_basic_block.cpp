#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * DVMBasicBlock
         */
        DVMBasicBlock::DVMBasicBlock(std::uint64_t start,
                                     std::shared_ptr<DalvikOpcodes> dalvik_opcodes,
                                     std::shared_ptr<BasicBlocks> context,
                                     std::shared_ptr<EncodedMethod> method,
                                     std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions)
        {
            this->start = start;
            this->end = end;
            this->dalvik_opcodes = dalvik_opcodes;
            this->context = context;
            this->method = method;
            this->instructions = instructions;

            // get name for basic block
            this->name = *this->method->get_method()->get_method_name();

            std::stringstream stream;
            stream << std::hex << this->start;

            this->name += "-BB@" + stream.str();
        }

        /**
         * @brief get start idx from the current basic block.
         * @return std::uint64_t
         */
        std::uint64_t DVMBasicBlock::get_start()
        {
            return start;
        }

        /**
         * @brief get end idx from the current basic block.
         * @return std::uint64_t
         */
        std::uint64_t DVMBasicBlock::get_end()
        {
            return end;
        }

        /**
         * @brief return all the instructions from current basic block.
         * @return std::vector<std::shared_ptr<Instruction>>
         */
        std::vector<std::shared_ptr<Instruction>> DVMBasicBlock::get_instructions()
        {
            std::vector<std::shared_ptr<Instruction>> bb_instructions;

            for (auto it = this->instructions.begin(); it != this->instructions.end(); it++)
            {
                if ((start <= it->first) && (it->first < end))
                    bb_instructions.push_back(it->second);
                else
                    break;
            }

            return bb_instructions;
        }

        /**
         * @brief return the last instruction from the basic block.
         * @return std::shared_ptr<Instruction>
         */
        std::shared_ptr<Instruction> DVMBasicBlock::get_last()
        {
            auto bb = get_instructions();
            return bb[bb.size() - 1];
        }

        /**
         * @brief return all the child basic blocks.
         * @return std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>>
         */
        std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> DVMBasicBlock::get_next()
        {
            return childs;
        }

        /**
         * @brief return all the parent basic blocks.
         * @return std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>>
         */
        std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> DVMBasicBlock::get_prev()
        {
            return parents;
        }

        /**
         * @brief push a basic block into the vector of parent basic blocks.
         * @param bb: std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>> to push in vector.
         * @return void
         */
        void DVMBasicBlock::set_parent(std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>> bb)
        {
            parents.push_back(bb);
        }

        /**
         * @brief set a children basic block, if no argument is given, this is taken from context.
         * @return void
         */
        void DVMBasicBlock::set_child()
        {
            auto next_block = context->get_basic_block_by_idx(end + 1);
            if (next_block != nullptr)
            {
                childs.push_back({end - last_length, end, next_block});
            }

            for (auto it = childs.begin(); it != childs.end(); it++)
            {
                if (std::get<2>(*it) != nullptr)
                    std::get<2>(*it)->set_parent({std::get<1>(*it), std::get<0>(*it), shared_from_this()});
            }
        }

        /**
         * @brief set a set of children basic blocks.
         * @param values: ids from context of basic blocks to push into vector.
         * @return void
         */
        void DVMBasicBlock::set_child(std::vector<int64_t> values)
        {
            for (auto it = values.begin(); it != values.end(); it++)
            {
                if (*it != -1)
                {
                    auto next_block = context->get_basic_block_by_idx(*it);
                    if (next_block != nullptr)
                    {
                        childs.push_back({end - last_length, *it, next_block});
                    }
                }
            }

            for (auto it = childs.begin(); it != childs.end(); it++)
            {
                if (std::get<2>(*it) != nullptr)
                    std::get<2>(*it)->set_parent({std::get<1>(*it), std::get<0>(*it), shared_from_this()});
            }
        }

        /**
         * @brief return last length of DVMBasicBlock.
         * @return std::uint64_t
         */
        std::uint64_t DVMBasicBlock::get_last_length()
        {
            return last_length;
        }

        /**
         * @brief return the number of instructions of the DVMBasicBlock.
         * @return std::uint64_t
         */
        std::uint64_t DVMBasicBlock::get_nb_instructions()
        {
            return nb_instructions;
        }

        /**
         * @brief Calculate new values with an instruction and push in case is a special instruction.
         * @param instr: std::shared_ptr<Instruction> object to increase diferent values and insert into special instructions.
         * @return void
         */
        void DVMBasicBlock::push(std::shared_ptr<Instruction> instr)
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

        /**
         * @brief get one of the special instructions.
         * @param idx: std::uint64_t with index of the special instruction.
         * @return std::shared_ptr<Instruction>
         */
        std::shared_ptr<Instruction> DVMBasicBlock::get_special_instruction(std::uint64_t idx)
        {
            if (special_instructions.find(idx) != special_instructions.end())
                return special_instructions[idx];
            else
                return nullptr;
        }

        /**
         * @brief return an exception analysis object.
         * @return std::shared_ptr<ExceptionAnalysis>
         */
        std::shared_ptr<ExceptionAnalysis> DVMBasicBlock::get_exception_analysis()
        {
            return exception_analysis;
        }

        /**
         * @brief set exception analysis object
         * @param exception_analysis: std::shared_ptr<ExceptionAnalysis> object.
         * @return void
         */
        void DVMBasicBlock::set_exception_analysis(std::shared_ptr<ExceptionAnalysis> exception_analysis)
        {
            this->exception_analysis = exception_analysis;
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

        /**
         * @brief push a given DVMBasicBlock into the vector.
         * @param basic_block: DVMBasicBlock object.
         * @return void
         */
        void BasicBlocks::push_basic_block(std::shared_ptr<DVMBasicBlock> basic_block)
        {
            this->basic_blocks.push_back(basic_block);
        }

        /**
         * @brief pop the last basic block from the vector, pop operation remove it from the vector.
         * @return std::shared_ptr<DVMBasicBlock>
         */
        std::shared_ptr<DVMBasicBlock> BasicBlocks::pop_basic_block()
        {
            std::shared_ptr<DVMBasicBlock> last_bb = nullptr;
            if (this->basic_blocks.size() >= 1)
            {
                last_bb = this->basic_blocks[this->basic_blocks.size() - 1];
                this->basic_blocks.pop_back();
            }

            return last_bb;
        }

        /**
         * @brief get one basic block by the idx of the instruction.
         * @param idx: index of the instruction to retrieve its basic block.
         * @return std::shared_ptr<DVMBasicBlock>
         */
        std::shared_ptr<DVMBasicBlock> BasicBlocks::get_basic_block_by_idx(std::uint64_t idx)
        {
            for (auto it = this->basic_blocks.begin(); it != this->basic_blocks.end(); it++)
            {
                if ((it->get()->get_start() <= idx) && (idx < it->get()->get_end()))
                {
                    return *it;
                }
            }

            return nullptr;
        }

        /**
         * @brief get the numbers of basic blocks.
         * @return size_t
         */
        size_t BasicBlocks::get_number_of_basic_blocks()
        {
            return basic_blocks.size();
        }

        /**
         * @brief get all the basic blocks.
         * @return std::vector<std::shared_ptr<DVMBasicBlock>>
         */
        std::vector<std::shared_ptr<DVMBasicBlock>> BasicBlocks::get_basic_blocks()
        {
            return basic_blocks;
        }

    }
}