#include "ir_blocks.hpp"

namespace KUNAI {
    namespace MJOLNIR {
        /**
         * IRBlock class
         */

        /**
         * @brief Constructor of IRBlock, this class represent blocks of statements.
         * @return void
         */
        IRBlock::IRBlock() {}

        /**
         * @brief Destructor of IRBlock, nothing to be done.
         * @return void
         */
        IRBlock::~IRBlock() {}

        /**
         * @brief Append one statement to the list, we use the class std::list as it allows us to
         *        insert instructions between other instructions easily.
         * @param statement: statement to append to the list.
         * @return void
         */
        void IRBlock::append_statement_to_block(std::shared_ptr<IRStmnt> statement)
        {
            block_statements.push_back(statement);
        }

        /**
         * @brief Add a block as one of the successors of the current block.
         * @param successor: successor block to add.
         * @return void
         */
        void IRBlock::add_block_to_sucessors(std::shared_ptr<IRBlock> successor)
        {
            successors.push_back(successor);
        }

        /**
         * @brief Add a block as on of the precessors of the current block.
         * @param predecessor: predecessor block to add.
         * @return void
         */
        void IRBlock::add_block_to_predecessors(std::shared_ptr<IRBlock> predecessor)
        {
            predecessors.push_back(predecessor);
        }

        /**
         * @brief Remove on of the statements given its position.
         * @param pos: position of the statement in the vector.
         * @return bool
         */
        bool IRBlock::delete_statement_by_position(size_t pos)
        {
            if (pos >= block_statements.size())
                return false;
                
            if (pos == 0)
            {
                auto first = block_statements.begin();
                // if delete the first one, set the next to NULL.
                block_statements.erase(first);
            }
            else if (pos == (block_statements.size() - 1))
            {
                auto last = std::prev(block_statements.end());

                block_statements.erase(last);
            }
            else
            {
                for (auto iter = block_statements.begin(); iter != block_statements.end(); iter++)
                {
                    if (pos-- == 0)
                    {
                        // erase iter
                        block_statements.erase(iter);
                        break;
                    }
                }
            }
            return true;
        }

        /**
         * @brief Remove one of the sucessors given an IRBlock, if it's inside of the vector.
         * @param block: Block to remove from the successors.
         * @return void
         */
        void IRBlock::delete_block_from_sucessors(std::shared_ptr<IRBlock> block)
        {
            auto position = std::find(successors.begin(), successors.end(), block);
            if (position != successors.end())
                successors.erase(position);
        }

        /**
         * @brief Remove on of the predecessors given an IRBlock, if it's inside of the vector.
         * @param block: Block to remove from the predecessors.
         * @return void
         */
        void IRBlock::delete_block_from_precessors(std::shared_ptr<IRBlock> block)
        {
            auto position = std::find(predecessors.begin(), predecessors.end(), block);
            if (position != predecessors.end())
                predecessors.erase(position);
        }

        /**
         * @brief Get the number of statements from the block.
         * @return size_t
         */
        size_t IRBlock::get_number_of_statements()
        {
            return block_statements.size();
        }

        /**
         * @brief Get the list of statements.
         * @return std::list<std::shared_ptr<IRStmnt>>
         */
        std::list<std::shared_ptr<IRStmnt>> IRBlock::get_statements()
        {
            return block_statements;
        }

        /**
         * @brief Get the number of successor blocks from the current block.
         * @return size_t
         */
        size_t IRBlock::get_number_of_successors()
        {
            return successors.size();
        }

        /**
         * @brief Get the list of successor blocks.
         * @return std::vector<std::shared_ptr<IRBlock>>
         */
        std::vector<std::shared_ptr<IRBlock>> IRBlock::get_successors()
        {
            return successors;
        }

        /**
         * @brief Get the number of predecessor blocks from the current block.
         * @return size_t
         */
        size_t IRBlock::get_number_of_predecessors()
        {
            return predecessors.size();
        }

        /**
         * @brief Get the list of predecessor blocks.
         * @return std::vector<std::shared_ptr<IRBlock>>
         */
        std::vector<std::shared_ptr<IRBlock>> IRBlock::get_predecessors()
        {
            return predecessors;
        }

        /**
         * @brief Get type of node depending on number of successors and predecessors.
         * @return node_type_t
         */
        IRBlock::node_type_t IRBlock::get_type_of_node()
        {
            if (successors.size() > 1)
                return BRANC_NODE;
            else if (predecessors.size() > 1)
                return JOIN_NODE;
            else
                return REGULAR_NODE;
        }
    }
}