#include "KUNAI/mjolnIR/ir_grammar.hpp"

namespace KUNAI {
    namespace MJOLNIR {
        /**
         * IRBlock class
         */
        IRBlock::IRBlock() {}

        
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

        
        std::string IRBlock::get_name()
        {
            std::stringstream stream;

            stream << "BB." << std::hex << start_idx << "->" << end_idx;

            return stream.str();
        }

        
        std::string IRBlock::to_string()
        {
            std::stringstream stream;

            stream << "[BB." << std::hex << start_idx << "->" << end_idx << "]\n";

            for (auto instr : block_statements)
                stream << instr->to_string() << "\n";
            
            return stream.str();
        }
    }
}