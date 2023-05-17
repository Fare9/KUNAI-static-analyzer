//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// 
// @file stream_exception.hpp
#ifndef KUNAI_EXCEPTIONS_STREAM_EXCEPTION_HPP
#define KUNAI_EXCEPTIONS_STREAM_EXCEPTION_HPP

#include <iostream>

namespace exceptions
{
    /// @brief Exception raised in stream error
    class StreamException : public std::exception
    {
        /// @brief message to show with the exception
        std::string _msg;

    public:
        
        /// @brief Constructor of exception
        /// @param msg message to show to the user
        StreamException(const std::string &msg) : _msg(msg)
        {}

        /// @brief Return error message
        /// @return error message in a c string style
        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
    };
} // namespace exceptions

#endif