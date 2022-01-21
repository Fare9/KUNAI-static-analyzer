#include "utils.hpp"


namespace KUNAI {

std::string read_ansii_string(std::istream& input_file, std::uint64_t offset)
{
    std::uint64_t current_offset = input_file.tellg();
    std::string new_str;
    std::uint8_t character = -1;

    // set offset of file in at the
    // given offset
    input_file.seekg(offset);
    std::uint32_t count = 0;

    while(character != 0)
    {
        // we have a maximum size for strings
        if (count == MAX_ANSII_STR_SIZE)
        {
            new_str = "";
            break;
        }
        input_file.read(reinterpret_cast<char*>(&character), sizeof(std::uint8_t));
        new_str += character;
        count++;
    }
    
    // now return again
    input_file.seekg(current_offset);
    return new_str;
}

std::string read_dex_string(std::istream& input_file, std::uint64_t offset)
{
    std::uint64_t current_offset = input_file.tellg();
    std::string new_str;
    std::uint8_t character = -1;
    std::uint64_t utf16_size;

    // set offset of file in at the
    // given offset
    input_file.seekg(offset);

    utf16_size = read_uleb128(input_file);

    while(utf16_size-- != 0)
    {
        input_file.read(reinterpret_cast<char*>(&character), sizeof(std::uint8_t));
        new_str += character;
    }

    // now return again
    input_file.seekg(current_offset);
    return new_str;
}

// From lief parser
std::uint64_t read_uleb128(std::istream& input_file)
{
    std::uint64_t value = 0;
    unsigned shift = 0;
    std::uint8_t byte_read, current;
    
    do
    {
        read_data_file<std::uint8_t>(byte_read, sizeof(std::uint8_t), input_file);
        value |= static_cast<std::uint64_t>(byte_read & 0x7f) << shift;
        shift += 7;
    } while (byte_read & 0x80);
    
    return value;
}

std::int64_t read_sleb128(std::istream& input_file)
{
    std::int64_t value = 0;
    unsigned shift = 0;
    std::uint8_t byte_read;
    do
    {
        read_data_file<std::uint8_t>(byte_read, sizeof(std::uint8_t), input_file);
        value |= static_cast<std::uint64_t>(byte_read & 0x7f) << shift;
        shift += 7;
    } while (byte_read & 0x80);
    // sign extend negative numbers
    if ((byte_read & 0x40) != 0)
        value |= static_cast<int64_t>(-1) << shift;

    return value;
}

}