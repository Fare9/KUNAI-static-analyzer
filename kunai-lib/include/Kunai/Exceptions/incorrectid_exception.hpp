//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// 
// @file incorrectid_exception.hpp
#ifndef KUNAI_EXCEPTIONS_INCORRECTID_EXCEPTION_HPP
#define KUNAI_EXCEPTIONS_INCORRECTID_EXCEPTION_HPP

#include <iostream>

namespace exceptions
{
    /// @brief Exception raised when found an incorrect ID of any type
    /// can be strings ids, type ids, method ids...
    class IncorrectIDException : public std::exception
    {
        /// @brief message to show with the exception
        std::string _msg;

    public:
        
        /// @brief Constructor of exception
        /// @param msg message to show to the user
        IncorrectIDException(const std::string &msg) : _msg(msg)
        {}

        /// @brief Return error message
        /// @return error message in a c string style
        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
    };
} // namespace exceptions

#endif // KUNAI_EXCEPTIONS_INCORRECTID_EXCEPTION_HPP