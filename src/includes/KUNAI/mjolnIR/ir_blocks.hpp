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
            enum node_type_t
            {
                JOIN_NODE = 0,  // type of node with len(predecessors) > 1
                BRANC_NODE,     // type of node with len(successors) > 1
                REGULAR_NODE,   // other cases
            };

            IRBlock();
            ~IRBlock();

            void append_statement_to_block(std::shared_ptr<IRStmnt> statement);
            void add_block_to_sucessors(std::shared_ptr<IRBlock> successor);
            void add_block_to_predecessors(std::shared_ptr<IRBlock> predecessor);


            bool delete_statement_by_position(size_t pos);
            void delete_block_from_sucessors(std::shared_ptr<IRBlock> block);
            void delete_block_from_precessors(std::shared_ptr<IRBlock> block);

            size_t get_number_of_statements();
            std::list<std::shared_ptr<IRStmnt>> get_statements();

            size_t get_number_of_successors();
            std::vector<std::shared_ptr<IRBlock>> get_successors();

            size_t get_number_of_predecessors();
            std::vector<std::shared_ptr<IRBlock>> get_predecessors();

            node_type_t get_type_of_node();
        private:
            //! statements from the basic block.
            std::list<std::shared_ptr<IRStmnt>> block_statements;
            //! successor blocks
            std::vector<std::shared_ptr<IRBlock>> successors;
            //! predecessor blocks
            std::vector<std::shared_ptr<IRBlock>> predecessors;
        };

    }
}