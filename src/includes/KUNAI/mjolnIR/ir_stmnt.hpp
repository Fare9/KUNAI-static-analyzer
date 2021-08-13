/**
 * @file ir_stmnt.hpp
 * @author Farenain
 * 
 * @brief Statement instructions are those that will be executed
 *        one after the other, these will change once we reach one
 *        instruction that change the control flow graph of the function
 *        or method, inside of the statements apart from the expressions
 *        we have those instructions that modify some flags or
 *        temporal registers that are used later in comparison in jumps
 *        to know if a jump is taken or not.
 */

#include <iostream>
#include <memory>
#include <vector>

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRStmnt
        {
        public:
            enum stmnt_type_t
            {
                UJMP_STMNT_T,
                CJMP_STMNT_T,
                RET_STMNT_T,
                EXPR_STMNT_T,
                NONE_STMNT_T = 99 // used to finish the chain of statements
            };

            IRStmnt();
            ~IRStmnt();

            void set_next_stmnt(std::shared_ptr<IRStmnt> next);
            std::shared_ptr<IRStmnt> get_next_stmnt();

        private:
            //! next statement in the code.
            std::shared_ptr<IRStmnt> next;
        };
    }
}