//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file logger.cpp

#include "Kunai/Utils/logger.hpp"
#include "Kunai/Exceptions/generic_exception.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

/// @brief global variable, set to the proper value
KUNAI::LOGGER::logger_output_t global_logger_output = KUNAI::LOGGER::logger_output_t::TO_STDERR;

/// @brief possible file name from user to write the logs
std::string log_file_name = "";

void KUNAI::LOGGER::LOG_TO_STDERR()
{
    global_logger_output = KUNAI::LOGGER::logger_output_t::TO_STDERR;
}

void KUNAI::LOGGER::LOG_TO_STDOUT()
{
    global_logger_output = KUNAI::LOGGER::logger_output_t::TO_CONSOLE;
}

void KUNAI::LOGGER::LOG_TO_FILE()
{
    global_logger_output = logger_output_t::TO_FILE;
}



spdlog::logger* KUNAI::LOGGER::logger()
{
    static std::shared_ptr<spdlog::logger> logger;

    if (logger != nullptr)
        return logger.get();

    switch (global_logger_output)
    {
    case KUNAI::LOGGER::logger_output_t::TO_CONSOLE:
        logger = spdlog::get("console");
        if (logger == nullptr)
            logger = spdlog::stdout_color_mt("console");
        break;
    case KUNAI::LOGGER::logger_output_t::TO_STDERR:
        logger = spdlog::get("stderr");
        if (logger == nullptr)
            logger = spdlog::stderr_color_mt("stderr");
        break;
    case KUNAI::LOGGER::logger_output_t::TO_FILE:
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

    return logger.get();
}
