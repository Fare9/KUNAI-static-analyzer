/**
 * @file lifter_android.hpp
 * @author Farenain
 *
 * @brief Class for lifting the code from the Android methods/classes to
 *        mjolnIR, different analysis can be applied in order to improve the
 *        representation of the IR.
 */

#ifndef LIFTER_ANDROID_HPP
#define LIFTER_ANDROID_HPP

#include <iostream>
#include <set>
// DEX
#include "DVM/dex_instructions.hpp"
#include "Analysis/dex_analysis.hpp"

// mjolnIR
#include "ir_graph.hpp"

// lifter
#include "lifter_android_instructions.hpp"

namespace KUNAI
{
    namespace LIFTER
    {
// Size in bits of types
#define NIBBLE_S 4
#define BYTE_S 8
#define WORD_S 16
#define DWORD_S 32
#define ADDR_S 32
#define QWORD_S 64

        class LifterAndroid
        {
        public:
            /**
             * @brief Constructor of lifter.
             */
            LifterAndroid();

            /**
             * @brief Lift a given method from a method_analysis object.
             *
             * @param method_analysis: method from Android to lift.
             * @param android_analysis: analysis from Android.
             * @return std::shared_ptr<MJOLNIR::IRGraph>
             */
            std::shared_ptr<MJOLNIR::IRGraph> lift_android_method(std::shared_ptr<DEX::MethodAnalysis> method_analysis, std::shared_ptr<DEX::Analysis> android_analysis);

            /**
             * @brief Lift an android basic block instructions to IR instructions.
             *
             * @param basic_block: basic block with Android instructions.
             * @param bb: IR Basic Block.
             * @return true
             * @return false
             */
            bool lift_android_basic_block(std::shared_ptr<DEX::DVMBasicBlock> basic_block, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief
             *
             * @param instruction: instruction from android to lift.
             * @param bb: IR Basic Block.
             * @return true
             * @return false
             */
            bool lift_android_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

        private:
            uint64_t temp_reg_id;
            uint64_t current_idx;

            /**
             * @brief Create a new IRReg with the id of the register,
             *        using different objects will be useful for having
             *        different registers in different methods.
             * @param reg_id: id of the register to create.
             * @return std::shared_ptr<MJOLNIR::IRReg>
             */
            std::shared_ptr<MJOLNIR::IRReg> make_android_register(std::uint32_t reg_id);

            /**
             * @brief Return a temporal register, used for operations like
             *        conditional jumps to have some place where to store result
             *        of comparison.
             *
             * @return std::shared_ptr<MJOLNIR::IRTempReg>
             */
            std::shared_ptr<MJOLNIR::IRTempReg> make_temporal_register();

            /**
             * @brief Create a NONE type useful for cases where one of
             *        the parameters does not exists.
             * @return std::shared_ptr<MJOLNIR::IRType>
             */
            std::shared_ptr<MJOLNIR::IRType> make_none_type();

            /**
             * @brief Create an Int type used for the DEX instructions
             * @param value: value of the integer as an unsigned of 64 bits
             * @param is_signed: boolean saying if the value must be signed or unsigned.
             * @param type_size: size of the integer type.
             * @return std::shared_ptr<MJOLNIR::IRConstInt>
             */
            std::shared_ptr<MJOLNIR::IRConstInt> make_int(std::uint64_t value, bool is_signed, size_t type_size);

            /**
             * @brief Generate a IRString type for Android, this will have the string
             *        and the size of the string.
             * @param value: string to generate the object.
             * @return std::shared_ptr<MJOLNIR::IRString>
             */
            std::shared_ptr<MJOLNIR::IRString> make_str(std::string value);

            /**
             * @brief Generate a IRClass type for Android, this will be nothing more
             *        than the complete name of the class.
             * @param value: class to generate the object.
             * @return std::shared_ptr<MJOLNIR::IRClass>
             */
            std::shared_ptr<MJOLNIR::IRClass> make_class(DEX::Class *value);

            /**
             * @brief Generate a IRField type for Android, this has the values from the
             *        FieldID.
             * @param field: FieldID to generate the object.
             * @return std::shared_ptr<MJOLNIR::IRField>
             */
            std::shared_ptr<MJOLNIR::IRField> make_field(DEX::FieldID *field);

            /**
             * @brief Generate a IRAssign instruction from the IR, this will
             *        be any instruction for assigning strings, classes or registers
             *        to registers.
             * @param instruction: instruction to lift.
             * @param bb: basic block where to insert the instruction.
             * @return void.
             */
            void lift_assignment_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate a IRBinOp instruction or IRUnaryOp instruction
             *        which represent any arithmetic logic instruction.
             * @param instruction: instruction to lift from arithmetic logic instructions.
             * @param bb: basic block where to insert the instructions.
             * @return void.
             */
            void lift_arithmetic_logic_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate a IRRet instruction which represent a ret instruction.
             *
             * @param instruction: instruction to lift.
             * @param bb: basic block where to insert the instructions.
             * @return void
             */
            void lift_ret_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate a IRBComp instruction which represent a comparison between two values.
             *
             * @param instruction: instruction to lift.
             * @param bb: basic block where to insert the instructions.
             * @return void
             */
            void lift_comparison_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate a IRCJmp instruction which represent a conditional jump.
             *
             * @param instruction: instruction to lift.
             * @param bb: basic block where to insert the instructions.
             * @return void
             */
            void lift_conditional_jump_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate a IRUJmp instruction which represent an unconditional jump.
             *
             * @param instruction
             * @param bb
             * @return void
             */
            void lift_unconditional_jump_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate the IRCall instruction which represent any kind of call function/method instruction.
             *
             * @param instruction
             * @param bb
             */
            void lift_call_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Set for a call instruction its return register.
             *        this will be just for those cases where the current
             *        instruction is some kind of move_result and then
             *        the previos instruction is a call.
             *
             * @param instruction
             * @param call
             */
            void lift_move_result_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRCall> call);

            /**
             * @brief Generate a IRLoad instruction, this will commonly go with an IRCast.
             *
             * @param instruction
             * @param bb
             */
            void lift_load_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Generate a IRStore instruction from different aput* instructions.
             *
             * @param instruction
             * @param bb
             */
            void lift_store_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Lift a NOP instruction.
             *
             * @param bb
             */
            void lift_nop_instructions(std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Lift a new instruction.
             *
             * @param instruction
             * @param bb
             */
            void lift_new_instructions(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            /**
             * @brief Lift those Android switch instructions.
             *
             * @param instruction
             * @param bb
             */
            void lift_switch_instructions(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb);

            // some utilities

            /**
             * @brief Fix for every jump instruction at the end of a basic block,
             *        its target and in case of conditional jump its fallthrough,
             *        this will give for each one, the basic block where it points
             *        to.
             *
             * @param bbs: basic blocks from the method.
             */
            void jump_target_analysis(std::vector<std::shared_ptr<KUNAI::DEX::DVMBasicBlock>> bbs);
            void fallthrough_target_analysis(std::shared_ptr<MJOLNIR::IRGraph> ir_graph);

            AndroidInstructions androidinstructions;
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

            //! lifted blocks
            std::map<std::shared_ptr<KUNAI::DEX::DVMBasicBlock>, std::shared_ptr<MJOLNIR::IRBlock>> lifted_blocks;

            //! Android analysis objecto to check internally
            std::shared_ptr<DEX::Analysis> android_analysis;
        };
    }
}

#endif