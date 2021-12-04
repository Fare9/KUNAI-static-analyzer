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
            /**
             * Assignment instructions
             */

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
                DEX::DVMTypes::Opcode::OP_SGET_SHORT,
                DEX::DVMTypes::Opcode::OP_SPUT,
                DEX::DVMTypes::Opcode::OP_SPUT_WIDE,
                DEX::DVMTypes::Opcode::OP_SPUT_OBJECT,
                DEX::DVMTypes::Opcode::OP_SPUT_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_SPUT_BYTE,
                DEX::DVMTypes::Opcode::OP_SPUT_CHAR,
                DEX::DVMTypes::Opcode::OP_SPUT_SHORT,
                DEX::DVMTypes::Opcode::OP_IGET,
                DEX::DVMTypes::Opcode::OP_IGET_WIDE,
                DEX::DVMTypes::Opcode::OP_IGET_OBJECT,
                DEX::DVMTypes::Opcode::OP_IGET_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_IGET_BYTE,
                DEX::DVMTypes::Opcode::OP_IGET_CHAR,
                DEX::DVMTypes::Opcode::OP_IGET_SHORT,
                DEX::DVMTypes::Opcode::OP_IPUT,
                DEX::DVMTypes::Opcode::OP_IPUT_WIDE,
                DEX::DVMTypes::Opcode::OP_IPUT_OBJECT,
                DEX::DVMTypes::Opcode::OP_IPUT_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_IPUT_BYTE,
                DEX::DVMTypes::Opcode::OP_IPUT_CHAR,
                DEX::DVMTypes::Opcode::OP_IPUT_SHORT,
            };

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
            // Assignment PUT instruction
            std::set<DEX::DVMTypes::Opcode> assigment_instruction21_put = {
                DEX::DVMTypes::Opcode::OP_SPUT,
                DEX::DVMTypes::Opcode::OP_SPUT_WIDE,
                DEX::DVMTypes::Opcode::OP_SPUT_OBJECT,
                DEX::DVMTypes::Opcode::OP_SPUT_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_SPUT_BYTE,
                DEX::DVMTypes::Opcode::OP_SPUT_CHAR,
                DEX::DVMTypes::Opcode::OP_SPUT_SHORT,
            };
            // Assignment instructions of type Instruction31c
            std::set<DEX::DVMTypes::Opcode> assigment_instruction31c = {
                DEX::DVMTypes::Opcode::OP_CONST_STRING_JUMBO};

            std::set<DEX::DVMTypes::Opcode> assignment_instruction22c_get = {
                DEX::DVMTypes::Opcode::OP_IGET,
                DEX::DVMTypes::Opcode::OP_IGET_WIDE,
                DEX::DVMTypes::Opcode::OP_IGET_OBJECT,
                DEX::DVMTypes::Opcode::OP_IGET_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_IGET_BYTE,
                DEX::DVMTypes::Opcode::OP_IGET_CHAR,
                DEX::DVMTypes::Opcode::OP_IGET_SHORT,
            };

            std::set<DEX::DVMTypes::Opcode> assignment_instruction22c_put = {
                DEX::DVMTypes::Opcode::OP_IPUT,
                DEX::DVMTypes::Opcode::OP_IPUT_WIDE,
                DEX::DVMTypes::Opcode::OP_IPUT_OBJECT,
                DEX::DVMTypes::Opcode::OP_IPUT_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_IPUT_BYTE,
                DEX::DVMTypes::Opcode::OP_IPUT_CHAR,
                DEX::DVMTypes::Opcode::OP_IPUT_SHORT,
            };

            /**
             * Arithmetic Logic instructions
             */
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
                DEX::DVMTypes::Opcode::OP_ADD_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_RSUB_INT,
                DEX::DVMTypes::Opcode::OP_MUL_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_DIV_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_REM_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_AND_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_OR_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_XOR_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_ADD_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_RSUB_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_MUL_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_DIV_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_REM_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_AND_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_OR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_XOR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_SHL_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_SHR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_USHR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_INT_TO_LONG,
                DEX::DVMTypes::Opcode::OP_INT_TO_FLOAT,
                DEX::DVMTypes::Opcode::OP_INT_TO_DOUBLE,
                DEX::DVMTypes::Opcode::OP_LONG_TO_INT,
                DEX::DVMTypes::Opcode::OP_LONG_TO_FLOAT,
                DEX::DVMTypes::Opcode::OP_LONG_TO_DOUBLE,
                DEX::DVMTypes::Opcode::OP_FLOAT_TO_INT,
                DEX::DVMTypes::Opcode::OP_FLOAT_TO_LONG,
                DEX::DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE,
                DEX::DVMTypes::Opcode::OP_DOUBLE_TO_INT,
                DEX::DVMTypes::Opcode::OP_DOUBLE_TO_LONG,
                DEX::DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT,
                DEX::DVMTypes::Opcode::OP_INT_TO_BYTE,
                DEX::DVMTypes::Opcode::OP_INT_TO_CHAR,
                DEX::DVMTypes::Opcode::OP_INT_TO_SHORT,
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
                DEX::DVMTypes::Opcode::OP_ADD_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_ADD_INT_LIT8,
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
                DEX::DVMTypes::Opcode::OP_RSUB_INT,
                DEX::DVMTypes::Opcode::OP_RSUB_INT_LIT8,
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
                DEX::DVMTypes::Opcode::OP_MUL_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_MUL_INT_LIT8,
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
                DEX::DVMTypes::Opcode::OP_DIV_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_DIV_INT_LIT8,
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
                DEX::DVMTypes::Opcode::OP_REM_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_REM_INT_LIT8,
            };

            // AND operations
            std::set<DEX::DVMTypes::Opcode> and_instruction = {
                DEX::DVMTypes::Opcode::OP_AND_INT,
                DEX::DVMTypes::Opcode::OP_AND_LONG,
                DEX::DVMTypes::Opcode::OP_AND_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_AND_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_AND_INT_LIT8,
            };

            // OR operations
            std::set<DEX::DVMTypes::Opcode> or_instruction = {
                DEX::DVMTypes::Opcode::OP_OR_INT,
                DEX::DVMTypes::Opcode::OP_OR_LONG,
                DEX::DVMTypes::Opcode::OP_OR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_OR_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_OR_INT_LIT8,
            };

            // XOR operations
            std::set<DEX::DVMTypes::Opcode> xor_instruction = {
                DEX::DVMTypes::Opcode::OP_XOR_INT,
                DEX::DVMTypes::Opcode::OP_XOR_LONG,
                DEX::DVMTypes::Opcode::OP_XOR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_XOR_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_XOR_INT_LIT8,
            };

            // SHL operations
            std::set<DEX::DVMTypes::Opcode> shl_instruction = {
                DEX::DVMTypes::Opcode::OP_SHL_INT,
                DEX::DVMTypes::Opcode::OP_SHL_LONG,
                DEX::DVMTypes::Opcode::OP_SHL_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHL_INT_LIT8,
            };

            // SHR operations
            std::set<DEX::DVMTypes::Opcode> shr_instruction = {
                DEX::DVMTypes::Opcode::OP_SHR_INT,
                DEX::DVMTypes::Opcode::OP_SHR_LONG,
                DEX::DVMTypes::Opcode::OP_SHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_SHR_INT_LIT8,
            };

            // USHR operations
            std::set<DEX::DVMTypes::Opcode> ushr_instruction = {
                DEX::DVMTypes::Opcode::OP_USHR_INT,
                DEX::DVMTypes::Opcode::OP_USHR_LONG,
                DEX::DVMTypes::Opcode::OP_USHR_INT_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_LONG_2ADDR,
                DEX::DVMTypes::Opcode::OP_USHR_INT_LIT8,
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

            std::set<DEX::DVMTypes::Opcode> instruction22s_binary_instruction = {
                DEX::DVMTypes::Opcode::OP_ADD_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_RSUB_INT,
                DEX::DVMTypes::Opcode::OP_MUL_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_DIV_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_REM_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_AND_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_OR_INT_LIT16,
                DEX::DVMTypes::Opcode::OP_XOR_INT_LIT16,
            };

            std::set<DEX::DVMTypes::Opcode> instruction22b_binary_instruction = {
                DEX::DVMTypes::Opcode::OP_ADD_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_RSUB_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_MUL_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_DIV_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_REM_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_AND_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_OR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_XOR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_SHL_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_SHR_INT_LIT8,
                DEX::DVMTypes::Opcode::OP_USHR_INT_LIT8,
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

            // Cast operations
            std::set<DEX::DVMTypes::Opcode> cast_instruction = {
                DEX::DVMTypes::Opcode::OP_INT_TO_LONG,
                DEX::DVMTypes::Opcode::OP_INT_TO_FLOAT,
                DEX::DVMTypes::Opcode::OP_INT_TO_DOUBLE,
                DEX::DVMTypes::Opcode::OP_LONG_TO_INT,
                DEX::DVMTypes::Opcode::OP_LONG_TO_FLOAT,
                DEX::DVMTypes::Opcode::OP_LONG_TO_DOUBLE,
                DEX::DVMTypes::Opcode::OP_FLOAT_TO_INT,
                DEX::DVMTypes::Opcode::OP_FLOAT_TO_LONG,
                DEX::DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE,
                DEX::DVMTypes::Opcode::OP_DOUBLE_TO_INT,
                DEX::DVMTypes::Opcode::OP_DOUBLE_TO_LONG,
                DEX::DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT,
                DEX::DVMTypes::Opcode::OP_INT_TO_BYTE,
                DEX::DVMTypes::Opcode::OP_INT_TO_CHAR,
                DEX::DVMTypes::Opcode::OP_INT_TO_SHORT,
            };

            /**
             * Return instructions
             */

            // Return operations
            std::set<DEX::DVMTypes::Opcode> ret_instruction = {
                DEX::DVMTypes::Opcode::OP_RETURN,
                DEX::DVMTypes::Opcode::OP_RETURN_WIDE,
                DEX::DVMTypes::Opcode::OP_RETURN_OBJECT,
                DEX::DVMTypes::Opcode::OP_RETURN_VOID,
            };

            /**
             * Comparison instructions
             */
            std::set<DEX::DVMTypes::Opcode> cmp_instruction = {
                DEX::DVMTypes::Opcode::OP_CMPL_FLOAT,
                DEX::DVMTypes::Opcode::OP_CMPG_FLOAT,
                DEX::DVMTypes::Opcode::OP_CMPL_DOUBLE,
                DEX::DVMTypes::Opcode::OP_CMPG_DOUBLE,
                DEX::DVMTypes::Opcode::OP_CMP_LONG};

            /**
             * Conditional jump instrucitons
             */
            std::set<DEX::DVMTypes::Opcode> jcc_instruction = {
                DEX::DVMTypes::Opcode::OP_IF_EQ,
                DEX::DVMTypes::Opcode::OP_IF_NE,
                DEX::DVMTypes::Opcode::OP_IF_LT,
                DEX::DVMTypes::Opcode::OP_IF_GE,
                DEX::DVMTypes::Opcode::OP_IF_GT,
                DEX::DVMTypes::Opcode::OP_IF_LE,
                DEX::DVMTypes::Opcode::OP_IF_EQZ,
                DEX::DVMTypes::Opcode::OP_IF_NEZ,
                DEX::DVMTypes::Opcode::OP_IF_LTZ,
                DEX::DVMTypes::Opcode::OP_IF_GEZ,
                DEX::DVMTypes::Opcode::OP_IF_GTZ,
                DEX::DVMTypes::Opcode::OP_IF_LEZ,
            };

            std::set<DEX::DVMTypes::Opcode> jcc_instruction22t = {
                DEX::DVMTypes::Opcode::OP_IF_EQ,
                DEX::DVMTypes::Opcode::OP_IF_NE,
                DEX::DVMTypes::Opcode::OP_IF_LT,
                DEX::DVMTypes::Opcode::OP_IF_GE,
                DEX::DVMTypes::Opcode::OP_IF_GT,
                DEX::DVMTypes::Opcode::OP_IF_LE,
            };

            std::set<DEX::DVMTypes::Opcode> jcc_instruction21t = {
                DEX::DVMTypes::Opcode::OP_IF_EQZ,
                DEX::DVMTypes::Opcode::OP_IF_NEZ,
                DEX::DVMTypes::Opcode::OP_IF_LTZ,
                DEX::DVMTypes::Opcode::OP_IF_GEZ,
                DEX::DVMTypes::Opcode::OP_IF_GTZ,
                DEX::DVMTypes::Opcode::OP_IF_LEZ,
            };

            /**
             * Unconditional jump instructions
             */
            std::set<DEX::DVMTypes::Opcode> jmp_instruction = {
                DEX::DVMTypes::Opcode::OP_GOTO,
                DEX::DVMTypes::Opcode::OP_GOTO_16,
                DEX::DVMTypes::Opcode::OP_GOTO_32,
            };

            /**
             * Call instructions (call to methods)
             */
            std::set<DEX::DVMTypes::Opcode> call_instructions = {
                DEX::DVMTypes::Opcode::OP_INVOKE_VIRTUAL,
                DEX::DVMTypes::Opcode::OP_INVOKE_SUPER,
                DEX::DVMTypes::Opcode::OP_INVOKE_DIRECT,
                DEX::DVMTypes::Opcode::OP_INVOKE_STATIC,
                DEX::DVMTypes::Opcode::OP_INVOKE_INTERFACE,
            };

            std::set<DEX::DVMTypes::Opcode> call_instruction35c = {
                DEX::DVMTypes::Opcode::OP_INVOKE_VIRTUAL,
                DEX::DVMTypes::Opcode::OP_INVOKE_SUPER,
                DEX::DVMTypes::Opcode::OP_INVOKE_DIRECT,
                DEX::DVMTypes::Opcode::OP_INVOKE_STATIC,
                DEX::DVMTypes::Opcode::OP_INVOKE_INTERFACE,
            };

            // move-result instructions
            std::set<DEX::DVMTypes::Opcode> move_result_instruction = {
                DEX::DVMTypes::Opcode::OP_MOVE_RESULT,
                DEX::DVMTypes::Opcode::OP_MOVE_RESULT_WIDE,
                DEX::DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT
            };
        
            /**
             * Load Instructions
             */
            std::set<DEX::DVMTypes::Opcode> load_instruction = {
                DEX::DVMTypes::Opcode::OP_AGET,
                DEX::DVMTypes::Opcode::OP_AGET_WIDE,
                DEX::DVMTypes::Opcode::OP_AGET_OBJECT,
                DEX::DVMTypes::Opcode::OP_AGET_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_AGET_BYTE,
                DEX::DVMTypes::Opcode::OP_AGET_CHAR,
                DEX::DVMTypes::Opcode::OP_AGET_SHORT,
            };

            /**
             * Store Instructions
             */
            std::set<DEX::DVMTypes::Opcode> store_instructions = {
                DEX::DVMTypes::Opcode::OP_APUT,
                DEX::DVMTypes::Opcode::OP_APUT_WIDE,
                DEX::DVMTypes::Opcode::OP_APUT_OBJECT,
                DEX::DVMTypes::Opcode::OP_APUT_BOOLEAN,
                DEX::DVMTypes::Opcode::OP_APUT_BYTE,
                DEX::DVMTypes::Opcode::OP_APUT_CHAR,
                DEX::DVMTypes::Opcode::OP_APUT_SHORT,
            };
        };
    }
}

#endif