//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// 
// @file outofbound_exception.hpp
#ifndef KUNAI_EXCEPTIONS_OUTOFBOUND_EXCEPTION_HPP
#define KUNAI_EXCEPTIONS_OUTOFBOUND_EXCEPTION_HPP

#include <iostream>

namespace exceptions
{
    /// @brief Exception raised when a read is done out
    /// of bound in case it is detected.
    class OutOfBoundException : public std::exception
    {
        /// @brief message to show with the exception
        std::string _msg;

    public:
        
        /// @brief Constructor of exception
        /// @param msg message to show to the user
        OutOfBoundException(const std::string &msg) : _msg(msg)
        {}

        /// @brief Return error message
        /// @return error message in a c string style
        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
    };
} // namespace exceptions

#endif // KUNAI_EXCEPTIONS_OUTOFBOUND_EXCEPTION_HPP