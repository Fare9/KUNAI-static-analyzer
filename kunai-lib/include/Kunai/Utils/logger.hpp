//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file logger.hpp
// @brief Manage all the logging information and classes.

#ifndef KUNAI_UTILS_LOGGER_HPP
#define KUNAI_UTILS_LOGGER_HPP

#include "Kunai/Exceptions/generic_exception.hpp"
#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

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

        /// @brief global variable, set to the proper value
        static logger_output_t global_logger_output = TO_STDERR;
        /// @brief possible file name from user to write the logs
        static std::string log_file_name = "";

        /// @brief Write the log to stderr
        void log_to_stderr()
        {
            global_logger_output = TO_STDERR;
        }

        /// @brief Write the log to console
        void log_to_console()
        {
            global_logger_output = TO_CONSOLE;
        }

        /// @brief Write to a file specified by user
        /// @param file_name file where to dump the log
        void log_to_file(std::string &file_name)
        {
            global_logger_output = TO_FILE;
            log_file_name = file_name;
        }

        /// @brief Method to retrieve a logger object, this object
        /// will be different depending on the type of logging
        /// required.
        /// @return
        inline std::shared_ptr<spdlog::logger> logger()
        {
            std::shared_ptr<spdlog::logger> logger;

            switch (global_logger_output)
            {
            case TO_CONSOLE:
                logger = spdlog::get("console");
                if (logger == nullptr)
                    logger = spdlog::stdout_color_mt("console");
                break;
            case TO_STDERR:
                logger = spdlog::get("stderr");
                if (logger == nullptr)
                    logger = spdlog::stderr_color_mt("stderr");
                break;
            case TO_FILE:
                logger = spdlog::get("file_logger");
                if (logger == nullptr)
                {
                    if (log_file_name.empty())
                        throw exceptions::generic_exception("logger(): log_file_name "
                                                            "provided is empty");
                    logger = spdlog::basic_logger_mt("file_logger", log_file_name);
                }
            default:
                throw exceptions::generic_exception("logger(): Option provided for "
                                                    "'global_logger_output' not valid");
            }

            return logger;
        }

    } // namespace LOGGER
} // namespace KUNAI

#endif // KUNAI_UTILS_LOGGER_HPP