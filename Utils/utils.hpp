#pragma once

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>

namespace KUNAI {

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
    std::uint64_t read_sleb128(std::istream& input_file);
}

#endif