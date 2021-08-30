/**
 * @file ir_blocks.hpp
 * @author Farenain
 * 
 * @brief Classes to manage the basic blocks from an IR method
 *        the basic blocks contain a vector with all its instructions
 *        probably we will be able to apply different analysis
 *        for creating the CFG, DFG, or sets like extended basic blocks.
 */


#include <iostream>
#include <map>
#include <vector>
#include <list>
#include <algorithm>
#include "ir_stmnt.hpp"
#include "exceptions.hpp"

namespace KUNAI {
    namespace MJOLNIR {
        
        class IRBlock
        {
        public:
            IRBlock();
            ~IRBlock();

            void append_statement_to_block(std::shared_ptr<IRStmnt> statement);

            bool delete_statement_by_position(size_t pos);

            size_t get_number_of_statements();
            std::list<std::shared_ptr<IRStmnt>> get_statements();
        private:
            //! statements from the basic block.
            std::list<std::shared_ptr<IRStmnt>> block_statements;
        };

    }
}