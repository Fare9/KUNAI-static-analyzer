//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file invalidinstruction_exception.hpp
#ifndef KUNAI_EXCEPTIONS_INVALIDINSTRUCTION_EXCEPTION_HPP
#define KUNAI_EXCEPTIONS_INVALIDINSTRUCTION_EXCEPTION_HPP

#include <iostream>

namespace exceptions
{
    /// @brief Exception raised when one of the instructions has
    /// not a valid format.
    class InvalidInstructionException : public std::exception
    {
        /// @brief message to show with the exception
        std::string _msg;
        /// @brief Instruction size for disassembler
        std::uint32_t _inst_size;
    public:
        
        /// @brief Constructor of exception
        /// @param msg message to show to the user
        /// @param inst_Size size of the instruction to skip that size
        InvalidInstructionException(const std::string &msg, std::uint32_t inst_size) 
            : _msg(msg), _inst_size(inst_size)
        {}

        /// @brief Return error message
        /// @return error message in a c string style
        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }

        /// @brief Get the value of
        /// @return 
        std::uint32_t get_inst_size() const
        {
            return _inst_size;
        }
    };
} // namespace exceptions

#endif // KUNAI_EXCEPTIONS_INVALIDINSTRUCTION_EXCEPTION_HPP