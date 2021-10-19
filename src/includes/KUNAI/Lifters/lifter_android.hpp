/**
 * @file lifter_android.hpp
 * @author Farenain
 * 
 * @brief Class for lifting the code from the Android methods/classes to
 *        mjolnIR, different analysis can be applied in order to improve the
 *        representation of the IR.
 */

#include <iostream>
#include <set>
// DEX
#include <DEX/DVM/dex_instructions.hpp>
#include <DEX/DVM/dex_dvm_types.hpp>

// mjolnIR
#include <mjolnIR/ir_type.hpp>
#include <mjolnIR/ir_expr.hpp>
#include <mjolnIR/ir_stmnt.hpp>
#include <mjolnIR/ir_blocks.hpp>
#include <mjolnIR/ir_graph.hpp>

namespace KUNAI 
{
    namespace LIFTER
    {

        #define NIBBLE 4
        #define BYTE 8
        #define WORD 16
        #define DWORD 32
        #define QWORD 64
        

        class LifterAndroid
        {
        public:
            std::shared_ptr<MJOLNIR::IRReg> make_android_register(std::uint32_t reg_id);
            std::shared_ptr<MJOLNIR::IRType> make_none_type();
            std::shared_ptr<MJOLNIR::IRConstInt> make_int(std::uint64_t value, bool is_signed, size_t type_size);
            std::shared_ptr<MJOLNIR::IRString> make_str(std::string value);
            std::shared_ptr<MJOLNIR::IRClass> make_class(DEX::Class* value);

            std::shared_ptr<MJOLNIR::IRExpr> lift_assignment_instruction(std::shared_ptr<DEX::Instruction> instruction);
            

        private:
            //! It is nice that while we are lifting to annotate
            //! possible values that a register holds, in that case
            //! we annotate it into a symbolic table and if later
            //! is needed its value, we retrieve it from this variable.
            std::map<std::uint32_t, std::shared_ptr<MJOLNIR::IRType>> symbolic_table;

            //! types can be created in a map for using them when necessary
            //! in the same block same registers, same strings or same fields
            //! can be used more than once.
            std::map<std::uint32_t, std::shared_ptr<MJOLNIR::IRReg>> regs;
            std::map<std::uint32_t, std::shared_ptr<MJOLNIR::IRString>> strings;

        };
    }
}