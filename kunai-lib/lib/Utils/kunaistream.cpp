//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file kunaistream.cpp
#include "Kunai/Utils/kunaistream.hpp"

using namespace KUNAI::stream;

void KunaiStream::initialize()
{
    // save previous pointer
    auto curr_pointer = input_file.tellg();

    // obtain the size
    input_file.seekg(0, std::ios::beg);
    auto fsize = input_file.tellg();
    input_file.seekg(0, std::ios::end);
    fsize = input_file.tellg() - fsize;
    // return to current pointer
    input_file.seekg(curr_pointer, std::ios::beg);

    file_size = static_cast<std::size_t>(fsize);
}

std::string KunaiStream::read_ansii_string(std::int64_t offset)
{
    std::string new_str;
    std::int8_t character = -1;
    auto int8_s = sizeof(std::int8_t);
    // as always save the current offset
    auto curr_offset = input_file.tellg();

    // set the offset
    input_file.seekg(static_cast<std::streampos>(offset));
    std::int32_t count = 0;

    while (character != 0 && count < MAX_ANSII_STR_SIZE)
    {
        input_file.read(reinterpret_cast<char *>(&character), int8_s);
        new_str += static_cast<char>(character);
        count++;
    }

    // return again
    input_file.seekg(curr_offset);

    if (count == MAX_ANSII_STR_SIZE)
        return "";

    return new_str;
}

std::string KunaiStream::read_dex_string(std::int64_t offset)
{
    std::string new_str;
    std::int8_t character = -1;
    std::uint64_t utf16_size;
    auto int8_s = sizeof(std::int8_t);
    // save current offset
    auto curr_offset = input_file.tellg();

    // set the offset to the given offset
    input_file.seekg(static_cast<std::streampos>(offset));

    utf16_size = read_uleb128();

    while (utf16_size-- > 0)
    {
        input_file.read(reinterpret_cast<char *>(&character), int8_s);
        new_str += static_cast<char>(character);
    }

    // return again
    input_file.seekg(curr_offset);
    return new_str;
}

std::uint64_t KunaiStream::read_uleb128()
{
    std::uint64_t value = 0;
    unsigned shift = 0;
    std::int8_t byte_read;

    do
    {
        read_data<std::int8_t>(byte_read, sizeof(std::int8_t));
        value |= static_cast<std::uint64_t>(byte_read & 0x7f) << shift;
        shift += 7;
    } while (byte_read & 0x80);

    return value;
}

std::int64_t KunaiStream::read_sleb128()
{
    std::int64_t value = 0;
    unsigned shift = 0;
    std::int8_t byte_read;

    do
    {
        read_data<std::int8_t>(byte_read, sizeof(std::int8_t));
        value |= static_cast<std::uint64_t>(byte_read & 0x7f) << shift;
        shift += 7;
    } while (byte_read & 0x80);

    // sign extend negative numbers
    if ((byte_read & 0x40))
        value |= static_cast<std::int64_t>(-1) << shift;
    
    return value;
}