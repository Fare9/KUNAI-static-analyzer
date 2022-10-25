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
#include "KUNAI/DEX/DVM/dex_instructions.hpp"
#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

// mjolnIR
#include "KUNAI/mjolnIR/Analysis/optimizer.hpp"

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
             * @return MJOLNIR::irgraph_t
             */
            MJOLNIR::irgraph_t lift_android_method(DEX::methodanalysis_t& method_analysis, DEX::analysis_t& android_analysis);

            /**
             * @brief Lift an android basic block instructions to IR instructions.
             *
             * @param basic_block: basic block with Android instructions.
             * @param bb: IR Basic Block.
             * @return true
             * @return false
             */
            bool lift_android_basic_block(DEX::dvmbasicblock_t& basic_block, MJOLNIR::irblock_t& bb);

            /**
             * @brief
             *
             * @param instruction: instruction from android to lift.
             * @param bb: IR Basic Block.
             * @return true
             * @return false
             */
            bool lift_android_instruction(DEX::instruction_t& instruction, MJOLNIR::irblock_t& bb);

        private:
            uint64_t temp_reg_id;
            uint64_t current_idx;


            /**
             * @brief Lift an instruction of type instruction23x, avoid repetition of code.
             * 
             * @param instruction: instruction to check.
             * @param bin_op: type of binary operation used.
             * @param cast_type: cast to correct type after instruction.
             * @param bb: basic block where to push the instructions.
             */
            void lift_instruction23x_binary_instruction(KUNAI::DEX::instruction_t& instruction,
                                                        KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                        MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                        MJOLNIR::irblock_t& bb);

            /**
             * @brief Lift an instruction of type instruction12x, avoid repetition of code.
             * 
             * @param instruction: instruction to check.
             * @param bin_op: type of binary operation used.
             * @param cast_type: cast to correct type after instruction.
             * @param bb: basic block where to push the instructions.
             */
            void lift_instruction12x_binary_instruction(KUNAI::DEX::instruction_t& instruction,
                                                        KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                        MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                        MJOLNIR::irblock_t& bb);

            /**
             * @brief Lift an instruction of type instruction22s
             * 
             * @param instruction 
             * @param bin_op 
             * @param cast_type 
             * @param bb 
             */
            void lift_instruction22s_binary_instruction(KUNAI::DEX::instruction_t& instruction,
                                                        KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                        MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                        MJOLNIR::irblock_t& bb);                

            /**
             * @brief Lift an instruction of type instruction22b
             * 
             * @param instruction 
             * @param bin_op 
             * @param cast_type 
             * @param bb 
             */
            void lift_instruction22b_binary_instruction(KUNAI::DEX::instruction_t& instruction,
                                                        KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                        MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                        MJOLNIR::irblock_t& bb);

            /**
             * @brief Lift an instruction of type instruction12x
             * 
             * @param instruction 
             * @param unary_op 
             * @param cast_type 
             * @param bb 
             */
            void lift_instruction12x_unary_instruction(KUNAI::DEX::instruction_t& instruction,
                                                        KUNAI::MJOLNIR::IRUnaryOp::unary_op_t unary_op,
                                                        MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                        MJOLNIR::irblock_t& bb);

            /**
             * @brief Lift a comparison instruction.
             * 
             * @param instruction 
             * @param cast_type 
             * @param comparison 
             * @param bb 
             */
            void lift_comparison_instruction(KUNAI::DEX::instruction_t& instruction,
                                             MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                             MJOLNIR::IRBComp::comp_t comparison,
                                             MJOLNIR::irblock_t& bb);

            /**
             * @brief Lift a conditional jump instruction of type instruction22t
             * 
             * @param instruction 
             * @param comparison 
             * @param bb 
             */
            void lift_jcc_instruction22t(KUNAI::DEX::instruction_t& instruction,
                                         MJOLNIR::IRBComp::comp_t comparison,
                                         MJOLNIR::irblock_t& bb);

            /**
             * @brief Lift a conditional jump instruction of type instruction21t (comparison against zero).
             * 
             * @param instruction 
             * @param comparison 
             * @param bb 
             */
            void lift_jcc_instruction21t(KUNAI::DEX::instruction_t& instruction,
                                         MJOLNIR::IRZComp::zero_comp_t comparison,
                                         MJOLNIR::irblock_t& bb);

            /**
             * @brief Generate a IRLoad instruction, this will commonly go with an IRCast.
             *
             * @param instruction
             * @param size
             * @param cast_type
             * @param bb
             */
            void lift_load_instruction(DEX::instruction_t instruction, size_t size, MJOLNIR::IRUnaryOp::cast_type_t cast_type, MJOLNIR::irblock_t bb);

            /**
             * @brief Generate a IRStore instruction.
             * 
             * @param instruction 
             * @param size 
             * @param bb 
             */
            void lift_store_instruction(DEX::instruction_t instruction, size_t size, MJOLNIR::irblock_t bb);

            /**
             * @brief Create a new IRReg with the id of the register,
             *        using different objects will be useful for having
             *        different registers in different methods.
             * @param reg_id: id of the register to create.
             * @return MJOLNIR::irreg_t
             */
            MJOLNIR::irreg_t make_android_register(std::uint32_t reg_id);

            /**
             * @brief Return a temporal register, used for operations like
             *        conditional jumps to have some place where to store result
             *        of comparison.
             *
             * @return MJOLNIR::irtempreg_t
             */
            MJOLNIR::irtempreg_t make_temporal_register();

            /**
             * @brief Create a NONE type useful for cases where one of
             *        the parameters does not exists.
             * @return MJOLNIR::irtype_t
             */
            MJOLNIR::irtype_t make_none_type();

            /**
             * @brief Create an Int type used for the DEX instructions
             * @param value: value of the integer as an unsigned of 64 bits
             * @param is_signed: boolean saying if the value must be signed or unsigned.
             * @param type_size: size of the integer type.
             * @return MJOLNIR::irconstint_t
             */
            MJOLNIR::irconstint_t make_int(std::uint64_t value, bool is_signed, size_t type_size);

            /**
             * @brief Generate a IRString type for Android, this will have the string
             *        and the size of the string.
             * @param value: string to generate the object.
             * @return MJOLNIR::irstring_t
             */
            MJOLNIR::irstring_t make_str(std::string value);

            /**
             * @brief Generate a IRClass type for Android, this will be nothing more
             *        than the complete name of the class.
             * @param value: class_t to generate the object.
             * @return MJOLNIR::irclass_t
             */
            MJOLNIR::irclass_t make_class(DEX::class_t value);

            /**
             * @brief Generate a IRFundamental type for Android, this will be nothing more
             *        than the name of the fundamental type
             * 
             * @param value fundamental_t to generate the object
             * @return MJOLNIR::irfundamental_t 
             */
            MJOLNIR::irfundamental_t make_fundamental(DEX::fundamental_t value);

            /**
             * @brief Generate a IRField type for Android, this has the values from the
             *        fieldid_t.
             * @param field: fieldid_t to generate the object.
             * @return MJOLNIR::irfield_t
             */
            MJOLNIR::irfield_t make_field(DEX::fieldid_t field);

            // some utilities

            /**
             * @brief Fix for every jump instruction at the end of a basic block,
             *        its target and in case of conditional jump its fallthrough,
             *        this will give for each one, the basic block where it points
             *        to.
             *
             * @param bbs: basic blocks from the method.
             * @param method_graph graph we are working on
             */
            void jump_target_analysis(std::vector<DEX::dvmbasicblock_t>& bbs, MJOLNIR::irgraph_t method_graph);

            //! lifted blocks
            std::map<DEX::DVMBasicBlock*, MJOLNIR::irblock_t> lifted_blocks;

            //! Android analysis object to check internally
            DEX::analysis_t android_analysis;

            //! Optimization passes
            MJOLNIR::optimizer_t optimizer;
        };
    }
}

#endif