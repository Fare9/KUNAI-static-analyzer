/**
 * @file lifter_android_instructions.hpp
 * @author Farenain
 * 
 * @brief Instructions from Android DALVIK for the lifter.
 */

#ifndef LIFTER_ANDROID_INSTRUCTIONS_HPP
#define LIFTER_ANDROID_INSTRUCTIONS_HPP

#include <set>
#include "DVM/dex_dvm_types.hpp"

namespace KUNAI
{
    namespace LIFTER
    {

        class AndroidInstructions
        {
        public:
            std::set<DEX::DVMTypes::Opcode> assignment_instruction = {
                DEX::DVMTypes::Opcode::OP_MOVE,
                DEX::DVMTypes::Opcode::OP_MOVE_WIDE,
                DEX::DVMTypes::Opcode::OP_MOVE_OBJECT,
                DEX::DVMTypes::Opcode::OP_MOVE_FROM16,
                DEX::DVMTypes::Opcode::OP_MOVE_WIDE_FROM16,
                DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16,
                DEX::DVMTypes::Opcode::OP_MOVE_16,
                DEX::DVMTypes::Opcode::OP_MOVE_WIDE_16,
                DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_16,
                //DEX::DVMTypes::Opcode::OP_MOVE_RESULT,
                //DEX::DVMTypes::Opcode::OP_MOVE_RESULT_WIDE,
                //DEX::DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT,
                //DEX::DVMTypes::Opcode::OP_MOVE_EXCEPTION,
                //DEX::DVMTypes::Opcode::OP_RETURN,
                //DEX::DVMTypes::Opcode::OP_RETURN_WIDE,
                //DEX::DVMTypes::Opcode::OP_RETURN_OBJECT,
                DEX::DVMTypes::Opcode::OP_CONST_4,
                DEX::DVMTypes::Opcode::OP_CONST_16,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE_16,
                DEX::DVMTypes::Opcode::OP_CONST,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE_32,
                DEX::DVMTypes::Opcode::OP_CONST_HIGH16,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE,
                DEX::DVMTypes::Opcode::OP_CONST_STRING,
                DEX::DVMTypes::Opcode::OP_CONST_CLASS,
                DEX::DVMTypes::Opcode::OP_CONST_STRING_JUMBO,
                DEX::DVMTypes::Opcode::OP_SGET,
                DEX::DVMTypes::Opcode::OP_SGET_WIDE,
                DEX::DVMTypes::Opcode::OP_SGET_OBJECT,
                DEX::DVMTypes::Opcode::OP_SGET_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_SGET_BYTE,
                DEX::DVMTypes::Opcode::OP_SGET_CHAR,
                DEX::DVMTypes::Opcode::OP_SGET_SHORT};

            // Assignment instructions of type Instruction12x
            std::set<DEX::DVMTypes::Opcode> assignment_instruction12x = {
                DEX::DVMTypes::Opcode::OP_MOVE,
                DEX::DVMTypes::Opcode::OP_MOVE_WIDE,
                DEX::DVMTypes::Opcode::OP_MOVE_OBJECT};
            // Assignment instructions of type Instruction22x
            std::set<DEX::DVMTypes::Opcode> assignment_instruction22x = {
                DEX::DVMTypes::Opcode::OP_MOVE_FROM16,
                DEX::DVMTypes::Opcode::OP_MOVE_WIDE_FROM16,
                DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16};
            // Assignment instructions of type Instruction32x
            std::set<DEX::DVMTypes::Opcode> assignment_instruction32x = {
                DEX::DVMTypes::Opcode::OP_MOVE_16,
                DEX::DVMTypes::Opcode::OP_MOVE_WIDE_16,
                DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_16};
            // Assignment instructions of type Instruction11x
            //std::set<DEX::DVMTypes::Opcode> assignment_instruction11x = {
            //    DEX::DVMTypes::Opcode::OP_MOVE_RESULT,
            //    DEX::DVMTypes::Opcode::OP_MOVE_RESULT_WIDE,
            //    DEX::DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT,
            //    DEX::DVMTypes::Opcode::OP_MOVE_EXCEPTION,
            //    DEX::DVMTypes::Opcode::OP_RETURN,
            //    DEX::DVMTypes::Opcode::OP_RETURN_WIDE,
            //    DEX::DVMTypes::Opcode::OP_RETURN_OBJECT};
            // Assignment instructions of type Instruction11n
            std::set<DEX::DVMTypes::Opcode> assigment_instruction11n = {
                DEX::DVMTypes::Opcode::OP_CONST_4};
            // Assignment instructions of type Instruction21s
            std::set<DEX::DVMTypes::Opcode> assigment_instruction21s = {
                DEX::DVMTypes::Opcode::OP_CONST_16,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE_16};
            // Assignment instructions of type Instruction31i
            std::set<DEX::DVMTypes::Opcode> assigment_instruction31i = {
                DEX::DVMTypes::Opcode::OP_CONST,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE_32};
            // Assignment instructions of type Instruction21h
            std::set<DEX::DVMTypes::Opcode> assigment_instruction21h = {
                DEX::DVMTypes::Opcode::OP_CONST_HIGH16,
                DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16};
            // Assignment instructions of type Instruction51l
            std::set<DEX::DVMTypes::Opcode> assigment_instruction51l = {
                DEX::DVMTypes::Opcode::OP_CONST_WIDE};
            // Assignment instructions of type Instruction21c
            std::set<DEX::DVMTypes::Opcode> assigment_instruction21c = {
                DEX::DVMTypes::Opcode::OP_CONST_STRING,
                DEX::DVMTypes::Opcode::OP_CONST_CLASS,
                DEX::DVMTypes::Opcode::OP_SGET,
                DEX::DVMTypes::Opcode::OP_SGET_WIDE,
                DEX::DVMTypes::Opcode::OP_SGET_OBJECT,
                DEX::DVMTypes::Opcode::OP_SGET_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_SGET_BYTE,
                DEX::DVMTypes::Opcode::OP_SGET_CHAR,
                DEX::DVMTypes::Opcode::OP_SGET_SHORT};
            // Assignment instructions of type Instruction31c
            std::set<DEX::DVMTypes::Opcode> assigment_instruction31c = {
                DEX::DVMTypes::Opcode::OP_CONST_STRING_JUMBO};

            std::set<DEX::DVMTypes::Opcode> arithmetic_logic_instruction = {
                DEX::DVMTypes::Opcode::OP_ADD_INT,
                DEX::DVMTypes::Opcode::OP_ADD_LONG,
                DEX::DVMTypes::Opcode::OP_ADD_FLOAT,
                DEX::DVMTypes::Opcode::OP_ADD_DOUBLE,
                DEX::DVMTypes::Opcode::OP_SUB_INT,
                DEX::DVMTypes::Opcode::OP_SUB_LONG,
                DEX::DVMTypes::Opcode::OP_SUB_FLOAT,
                DEX::DVMTypes::Opcode::OP_SUB_DOUBLE,
                DEX::DVMTypes::Opcode::OP_MUL_INT,
                DEX::DVMTypes::Opcode::OP_MUL_LONG,
                DEX::DVMTypes::Opcode::OP_MUL_FLOAT,
                DEX::DVMTypes::Opcode::OP_MUL_DOUBLE,
                DEX::DVMTypes::Opcode::OP_DIV_INT,
                DEX::DVMTypes::Opcode::OP_DIV_LONG,
                DEX::DVMTypes::Opcode::OP_DIV_FLOAT,
                DEX::DVMTypes::Opcode::OP_DIV_DOUBLE,
                DEX::DVMTypes::Opcode::OP_REM_INT,
                DEX::DVMTypes::Opcode::OP_REM_LONG,
                DEX::DVMTypes::Opcode::OP_REM_FLOAT,
                DEX::DVMTypes::Opcode::OP_REM_DOUBLE,
                DEX::DVMTypes::Opcode::OP_AND_INT,
                DEX::DVMTypes::Opcode::OP_AND_LONG,
                DEX::DVMTypes::Opcode::OP_OR_INT,
                DEX::DVMTypes::Opcode::OP_OR_LONG,
                DEX::DVMTypes::Opcode::OP_XOR_INT,
                DEX::DVMTypes::Opcode::OP_XOR_LONG,
                DEX::DVMTypes::Opcode::OP_SHL_INT,
                DEX::DVMTypes::Opcode::OP_SHL_LONG,
                DEX::DVMTypes::Opcode::OP_SHR_INT,
                DEX::DVMTypes::Opcode::OP_SHR_LONG,
                DEX::DVMTypes::Opcode::OP_USHR_INT,
                DEX::DVMTypes::Opcode::OP_USHR_LONG,
                DEX::DVMTypes::Opcode::OP_ADD_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_ADD_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_NEG_INT,
                DEX::DVMTypes::Opcode::OP_NEG_LONG,
                DEX::DVMTypes::Opcode::OP_NEG_FLOAT,
                DEX::DVMTypes::Opcode::OP_NEG_DOUBLE,
                DEX::DVMTypes::Opcode::OP_NOT_INT,
                DEX::DVMTypes::Opcode::OP_NOT_LONG,
            };

            // ADD Operations
            std::set<DEX::DVMTypes::Opcode> add_instruction = {
                DEX::DVMTypes::Opcode::OP_ADD_INT,
                DEX::DVMTypes::Opcode::OP_ADD_LONG,
                DEX::DVMTypes::Opcode::OP_ADD_FLOAT,
                DEX::DVMTypes::Opcode::OP_ADD_DOUBLE,
                DEX::DVMTypes::Opcode::OP_ADD_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_ADD_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR,
            };

            // SUB operations
            std::set<DEX::DVMTypes::Opcode> sub_instruction = {
                DEX::DVMTypes::Opcode::OP_SUB_INT,
                DEX::DVMTypes::Opcode::OP_SUB_LONG,
                DEX::DVMTypes::Opcode::OP_SUB_FLOAT,
                DEX::DVMTypes::Opcode::OP_SUB_DOUBLE,
                DEX::DVMTypes::Opcode::OP_SUB_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR,
            };

            // MUL operations
            std::set<DEX::DVMTypes::Opcode> mul_instruction = {
                DEX::DVMTypes::Opcode::OP_MUL_INT,
                DEX::DVMTypes::Opcode::OP_MUL_LONG,
                DEX::DVMTypes::Opcode::OP_MUL_FLOAT,
                DEX::DVMTypes::Opcode::OP_MUL_DOUBLE,
                DEX::DVMTypes::Opcode::OP_MUL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR,
            };

            // DIV operations
            std::set<DEX::DVMTypes::Opcode> div_instruction = {
                DEX::DVMTypes::Opcode::OP_DIV_INT,
                DEX::DVMTypes::Opcode::OP_DIV_LONG,
                DEX::DVMTypes::Opcode::OP_DIV_FLOAT,
                DEX::DVMTypes::Opcode::OP_DIV_DOUBLE,
                DEX::DVMTypes::Opcode::OP_DIV_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR,
            };

            // Module operations
            std::set<DEX::DVMTypes::Opcode> mod_instruction = {
                DEX::DVMTypes::Opcode::OP_REM_INT,
                DEX::DVMTypes::Opcode::OP_REM_LONG,
                DEX::DVMTypes::Opcode::OP_REM_FLOAT,
                DEX::DVMTypes::Opcode::OP_REM_DOUBLE,
                DEX::DVMTypes::Opcode::OP_REM_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR,
            };

            // AND operations
            std::set<DEX::DVMTypes::Opcode> and_instruction = {
                DEX::DVMTypes::Opcode::OP_AND_INT,
                DEX::DVMTypes::Opcode::OP_AND_LONG,
                DEX::DVMTypes::Opcode::OP_AND_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_LONG_2ADDR,
            };

            // OR operations
            std::set<DEX::DVMTypes::Opcode> or_instruction = {
                DEX::DVMTypes::Opcode::OP_OR_INT,
                DEX::DVMTypes::Opcode::OP_OR_LONG,
                DEX::DVMTypes::Opcode::OP_OR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_LONG_2ADDR,
            };

            // XOR operations
            std::set<DEX::DVMTypes::Opcode> xor_instruction = {
                DEX::DVMTypes::Opcode::OP_XOR_INT,
                DEX::DVMTypes::Opcode::OP_XOR_LONG,
                DEX::DVMTypes::Opcode::OP_XOR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_LONG_2ADDR,
            };

            // SHL operations
            std::set<DEX::DVMTypes::Opcode> shl_instruction = {
                DEX::DVMTypes::Opcode::OP_SHL_INT,
                DEX::DVMTypes::Opcode::OP_SHL_LONG,
                DEX::DVMTypes::Opcode::OP_SHL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_LONG_2ADDR,
            };

            // SHR operations
            std::set<DEX::DVMTypes::Opcode> shr_instruction = {
                DEX::DVMTypes::Opcode::OP_SHR_INT,
                DEX::DVMTypes::Opcode::OP_SHR_LONG,
                DEX::DVMTypes::Opcode::OP_SHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_LONG_2ADDR,
            };

            // USHR operations
            std::set<DEX::DVMTypes::Opcode> ushr_instruction = {
                DEX::DVMTypes::Opcode::OP_USHR_INT,
                DEX::DVMTypes::Opcode::OP_USHR_LONG,
                DEX::DVMTypes::Opcode::OP_USHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_LONG_2ADDR,
            };

            std::set<DEX::DVMTypes::Opcode> instruction23x_binary_instruction = {
                DEX::DVMTypes::Opcode::OP_ADD_INT,
                DEX::DVMTypes::Opcode::OP_ADD_LONG,
                DEX::DVMTypes::Opcode::OP_ADD_FLOAT,
                DEX::DVMTypes::Opcode::OP_ADD_DOUBLE,
                DEX::DVMTypes::Opcode::OP_SUB_INT,
                DEX::DVMTypes::Opcode::OP_SUB_LONG,
                DEX::DVMTypes::Opcode::OP_SUB_FLOAT,
                DEX::DVMTypes::Opcode::OP_SUB_DOUBLE,
                DEX::DVMTypes::Opcode::OP_MUL_INT,
                DEX::DVMTypes::Opcode::OP_MUL_LONG,
                DEX::DVMTypes::Opcode::OP_MUL_FLOAT,
                DEX::DVMTypes::Opcode::OP_MUL_DOUBLE,
                DEX::DVMTypes::Opcode::OP_DIV_INT,
                DEX::DVMTypes::Opcode::OP_DIV_LONG,
                DEX::DVMTypes::Opcode::OP_DIV_FLOAT,
                DEX::DVMTypes::Opcode::OP_DIV_DOUBLE,
                DEX::DVMTypes::Opcode::OP_REM_INT,
                DEX::DVMTypes::Opcode::OP_REM_LONG,
                DEX::DVMTypes::Opcode::OP_REM_FLOAT,
                DEX::DVMTypes::Opcode::OP_REM_DOUBLE,
                DEX::DVMTypes::Opcode::OP_AND_INT,
                DEX::DVMTypes::Opcode::OP_AND_LONG,
                DEX::DVMTypes::Opcode::OP_OR_INT,
                DEX::DVMTypes::Opcode::OP_OR_LONG,
                DEX::DVMTypes::Opcode::OP_XOR_INT,
                DEX::DVMTypes::Opcode::OP_XOR_LONG,
                DEX::DVMTypes::Opcode::OP_SHL_INT,
                DEX::DVMTypes::Opcode::OP_SHL_LONG,
                DEX::DVMTypes::Opcode::OP_SHR_INT,
                DEX::DVMTypes::Opcode::OP_SHR_LONG,
                DEX::DVMTypes::Opcode::OP_USHR_INT,
                DEX::DVMTypes::Opcode::OP_USHR_LONG,
            };

            std::set<DEX::DVMTypes::Opcode> instruction12x_binary_instruction = {
                DEX::DVMTypes::Opcode::OP_ADD_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_INT_2ADDR,

                DEX::DVMTypes::Opcode::OP_ADD_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_LONG_2ADDR,

                DEX::DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_FLOAT_2ADDR,

                DEX::DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR,
                DEX::DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR,
            };

            // NEG operations
            std::set<DEX::DVMTypes::Opcode> neg_instruction = {
                DEX::DVMTypes::Opcode::OP_NEG_INT,
                DEX::DVMTypes::Opcode::OP_NEG_LONG,
                DEX::DVMTypes::Opcode::OP_NEG_FLOAT,
                DEX::DVMTypes::Opcode::OP_NEG_DOUBLE,
            };

            // NOT operations
            std::set<DEX::DVMTypes::Opcode> not_instruction = {
                DEX::DVMTypes::Opcode::OP_NOT_INT,
                DEX::DVMTypes::Opcode::OP_NOT_LONG,
            };

            std::set<DEX::DVMTypes::Opcode> instruction12x_unary_instruction = {
                DEX::DVMTypes::Opcode::OP_NEG_INT,
                DEX::DVMTypes::Opcode::OP_NEG_LONG,
                DEX::DVMTypes::Opcode::OP_NEG_FLOAT,
                DEX::DVMTypes::Opcode::OP_NEG_DOUBLE,
                DEX::DVMTypes::Opcode::OP_NOT_INT,
                DEX::DVMTypes::Opcode::OP_NOT_LONG,
            };
        };
    }
}

#endif