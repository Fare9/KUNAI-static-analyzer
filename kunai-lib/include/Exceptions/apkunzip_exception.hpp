//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// 
// @file apkunzip_exception.hpp
#ifndef EXCEPTIONS_APKUNZIP_EXCEPTION_HPP
#define EXCEPTIONS_APKUNZIP_EXCEPTION_HPP

#include <iostream>

namespace exceptions
{
    /// @brief Exception raised when unzipping process fails.
    class ApkUnzipException : public std::exception
    {
        /// @brief message to show with the exception
        std::string _msg;

    public:
        
        /// @brief Constructor of exception
        /// @param msg message to show to the user
        ApkUnzipException(const std::string &msg) : _msg(msg)
        {}

        /// @brief Return error message
        /// @return error message in a c string style
        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
    };
} // namespace exceptions



#endif // EXCEPTIONS_APKUNZIP_EXCEPTION_HPP