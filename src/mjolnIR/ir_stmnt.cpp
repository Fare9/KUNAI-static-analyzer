#include "ir_stmnt.hpp"


namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * IRStmnt class
         */

        /**
         * @brief Constructor of IRStmnt, really nothing to be done here.
         * @return void
         */
        IRStmnt::IRStmnt() {}

        /**
         * @brief Destructor of IRStmnt, nothing to be done.
         * @return void
         */
        IRStmnt::~IRStmnt() {}

        /**
         * @brief Set the next statement in the queue to analyze.
         * @param next: next statement for the analysis.
         * @return void
         */
        void IRStmnt::set_next_stmnt(std::shared_ptr<IRStmnt> next)
        {
            this->next = next;
        }

        /**
         * @brief Get the next statement in the queue to analyze.
         * @return std::shared_ptr<IRStmnt>
         */
        std::shared_ptr<IRStmnt> IRStmnt::get_next_stmnt()
        {
            return next;
        }
        
    }
}