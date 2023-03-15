//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file disassembler.hpp
// @brief Utilities of the disassembler to use by the different disassembly
// algorithms.

#ifndef KUNAI_DEX_DVM_DISASSEMBLER_HPP
#define KUNAI_DEX_DVM_DISASSEMBLER_HPP

#include "Kunai/DEX/DVM/dalvik_instructions.hpp"

#include <memory>
#include <iostream>
#include <any>

#include <unordered_map>
#include <map>
#include <utility>


namespace KUNAI
{
namespace DEX
{

    class Disassembler
    {
    public:
        /// @brief Information for the handler
        /// of exceptions, handler type, the
        /// start address of it and basic blocks
        typedef struct _handler_data
        {
            DVMType*                handler_type;
            std::uint64_t           handler_start_addr;
        } handler_data;

        /// @brief Information for the exceptions in the
        /// code
        typedef struct _exceptions_data
        {
            std::uint64_t               try_value_start_addr;
            std::uint64_t               try_value_end_addr;
            std::vector<handler_data>   handler;
        } exceptions_data;

        /// @brief For those who just want the full set of instructions
        /// it is possible to retrieve a vector with all the instructions
        /// from the method, it is not needed that these are sorted in any
        /// way                                         
        using instructions_t = std::unordered_map
                                <EncodedMethod*,
                                std::vector<std::unique_ptr<Instruction>>>;

    private:
        /// @brief pointer to the parser of the DEX file
        Parser * parser;
        /// @brief pointer to the last instruction generated
        /// by the Disassembler
        Instruction * last_instr;
        /// @brief In case there are no handlers, create a throwable one
        DVMClass throwable_class{"Ljava/lang/Throwable;"};
        
    public:

        /// @brief Constructor of the internal Disassembler for Dalvik
        Disassembler() = default;

        /// @brief Set the parser for the disassembler
        /// @param parser parser to use in the disassembler
        void set_parser(Parser * parser)
        {
            this->parser = parser;
        }

        /// @brief Get an instruction object from the op
        /// @param opcode op code of the instruction to return 
        /// @param bytecode reference to the bytecode for disassembly
        /// @param index index of the current instruction to analyze
        /// @return unique pointer to the disassembled Instruction
        std::unique_ptr<Instruction> disassemble_instruction(
            std::uint32_t opcode,
            std::vector<uint8_t> & bytecode,
            std::size_t index
        );

        /// @brief Determine given the last instruction the next instruction
        /// to run, the bytecode is retrieved from a :class:EncodedMethod.
        /// The offsets are calculated in number of bytes from the start of the
        /// method. Note, the offsets inside the bytecode are denoted in 16 bits
        /// units but method returns actual byte offsets.
        /// @param instruction instruction to obtain the next instructions
        /// @param curr_idx Current idx to calculate the newer one
        /// @return list of different offsets where code can go after the current
        /// instruction. Instructions like `if` or `switch` have more than one
        /// target, but `throw`, `return` and `goto` have just one. If entered
        /// opcode is not a branch instruction, next instruction is returned.
        std::vector<std::int64_t> determine_next(Instruction * instruction, std::uint64_t curr_idx);
    
        /// @brief Same as the other `determine_next` but the instruction we give
        /// is the instruction `last_instr` that Disassembler stores.
        /// @param curr_idx Current idx to calculate the newer one
        /// @return list of different offsets where code can go after the current
        /// instruction. Instructions like `if` or `switch` have more than one
        /// target, but `throw`, `return` and `goto` have just one. If entered
        /// opcode is not a branch instruction, next instruction is returned.
        std::vector<std::int64_t> determine_next(std::uint64_t curr_idx);

        /// @brief Given an instruction check if it is a conditional jump
        /// and retrieve in that case the target of the jump
        /// @param instr instruction to retrieve the target of the jump
        /// @return target of a conditional jump
        std::int16_t get_conditional_jump_target(Instruction * instr);

        /// @brief Given an instruction check if it is an unconditional jump
        /// and retrieve in that case the target of the jump
        /// @param instr instruction to retrieve the target of the jump
        /// @return target of an unconditional jump
        std::int32_t get_unconditional_jump_target(Instruction * instr);

        /// @brief Retrieve information from possible exception code inside
        /// of a method
        /// @param method method to extract exception data
        /// @return exception data in a vector
        std::vector<exceptions_data> determine_exception(EncodedMethod * method);
    };
} // namespace DEX
} // namespace KUNAI


#endif