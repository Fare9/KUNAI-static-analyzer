//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file logger.hpp
// @brief Manage all the logging information and classes.

#ifndef KUNAI_UTILS_LOGGER_HPP
#define KUNAI_UTILS_LOGGER_HPP


#include <iostream>
#include <spdlog/spdlog.h>

namespace KUNAI
{
namespace LOGGER
{
    /// @brief Output where to drop the logging from
    /// Kunai
    enum logger_output_t : std::uint8_t
    {
        TO_CONSOLE = 0, /// stdout
        TO_STDERR,      /// stderr
        TO_FILE         /// given file
    };

    void LOG_TO_STDERR();

    void LOG_TO_STDOUT();

    void LOG_TO_FILE();
    
    /// @brief Method to retrieve a logger object, this object
    /// will be different depending on the type of logging
    /// required.
    /// @return logger shared object
    spdlog::logger* logger();
        
} // namespace LOGGER
} // namespace KUNAI

#endif // KUNAI_UTILS_LOGGER_HPP