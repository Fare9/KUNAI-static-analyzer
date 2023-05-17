//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// 
// @file generic_exception.hpp
#ifndef KUNAI_EXCEPTIONS_GENERIC_EXCEPTION_HH
#define KUNAI_EXCEPTIONS_GENERIC_EXCEPTION_HH

#include <iostream>

namespace exceptions
{
    /// @brief A generic exception that can be used for common errors out
    /// of an internal from Kunai
    class generic_exception : public std::exception
    {
        /// @brief message to show with the exception
        std::string _msg;

    public:
        
        /// @brief Constructor of exception
        /// @param msg message to show to the user
        generic_exception(const std::string &msg) : _msg(msg)
        {}

        /// @brief Return error message
        /// @return error message in a c string style
        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
    };
}
#endif // KUNAI_EXCEPTIONS_GENERIC_EXCEPTION_HH