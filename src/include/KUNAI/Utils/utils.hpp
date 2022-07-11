#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace KUNAI {

    namespace LOGGER {
        enum logger_output_t {
            TO_CONSOLE = 0,
            TO_STDERR = 1,
            TO_FILE = 2
        };

        logger_output_t static global_logger_output = TO_STDERR;
        static std::string log_file_name = "";

        inline std::shared_ptr<spdlog::logger> logger()
        {
            switch (global_logger_output)
            {
            case TO_CONSOLE:
            {
                auto console_logger = spdlog::get("console");
                if (console_logger == nullptr)
                    return spdlog::stdout_color_mt("console");
                else
                    return console_logger;
            }
                break;
            case TO_STDERR:
            {
                auto error_logger = spdlog::get("stderr");
                if (error_logger == nullptr)
                    return spdlog::stderr_color_mt("stderr");
                else
                    return error_logger;
            }
                break;
            case TO_FILE:
            {
                auto file_logger = spdlog::get("file_logger");
                if (file_logger == nullptr)
                    return spdlog::basic_logger_mt("file_logger", log_file_name);
                else
                    return file_logger;
            }
                break;
            default:
                break;
            }

            return nullptr;
        }
    }

    const std::uint32_t MAX_ANSII_STR_SIZE = 256;

    template <typename table>
    bool read_data_file(table& file_table, std::uint32_t read_size, std::istream& input_file)
    /**
    *   Read file using a template to read specific structure
    *   or type.
    *   
    *   :param file_table: buffer where to read the data.
    *   :param read_size: size to read.
    *   :param input_file: file where to read from.
    */
    {
        if (!input_file)
            return false;

        input_file.read(reinterpret_cast<char *>(&file_table), read_size);

        if (input_file)
            return true;
        else
            return false;
    }
    
    std::string read_ansii_string(std::istream& input_file, std::uint64_t offset);
    std::string read_dex_string(std::istream& input_file, std::uint64_t offset);
    std::uint64_t read_uleb128(std::istream& input_file);
    std::int64_t read_sleb128(std::istream& input_file);
}

#endif