#include "ir_grammar.hpp"

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
    }
}