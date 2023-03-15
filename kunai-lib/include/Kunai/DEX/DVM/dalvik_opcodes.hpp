//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dalvik_opcodes.hpp
// @brief Utilities for managing opcodes from the dalvik machine

#ifndef KUNAI_DEX_DVM_DALVIK_OPCODES_HPP
#define KUNAI_DEX_DVM_DALVIK_OPCODES_HPP

#include "Kunai/DEX/parser/encoded.hpp"
#include "Kunai/DEX/DVM/dvm_types.hpp"
#include <iostream>

namespace KUNAI
{
namespace DEX
{
    /// @brief Class with static functions to manage information
    /// about the dalvik opcodes
    class DalvikOpcodes
    {
    public:
        /// @brief Constructor of Dalvik Opcodes
        DalvikOpcodes() = default;
        /// @brief Destructor of Dalvik Opcodes
        ~DalvikOpcodes() = default;

        /// @brief Find the instruction opcode in a map to obtain the
        /// instruction name
        /// @param instruction opcode of the instruction
        /// @return reference to instruction name
        static const std::string& get_instruction_name(std::uint32_t instruction);
        /// @brief Find the instruction type given an instruction opcode
        /// @param instruction opcode of the instruction
        /// @return kind of a instruction
        static TYPES::Kind get_instruction_type(std::uint32_t instruction);
        /// @brief Get an instruction type given an instruction opcode in string format
        /// @param instruction 
        /// @return constant reference to string with the Kind
        static const std::string& get_instruction_type_string(std::uint32_t instruction);
        /// @brief Find the instruction operation given an instruction opcode
        /// @param instruction opcode of the instruction
        /// @return operation of a instruction
        static TYPES::Operation get_instruction_operation(std::uint32_t instruction);
        /// @brief Get a string representation of the access flags from a method
        /// @param method method to retrieve its access flags
        /// @return string representation of access flags
        static std::string get_method_access_flags(EncodedMethod* method);
        /// @brief Get a string representation of the access flags from a field
        /// @param method field to retrieve its access flags
        /// @return string representation of access flags
        static std::string get_field_access_flags(EncodedField* field);
        /// @brief Get a string representation of any access flags
        /// @param access_flags access flags value
        /// @return string representation of access flags
        static std::string get_access_flags_str(TYPES::access_flags flags);
    };
} // namespace DEX
} // namespace KUNAI


#endif

