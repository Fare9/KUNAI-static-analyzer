//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file strings.hpp
// @brief Class for managing the list of strings, these strings
// follow an specific format where a `string_data_off` is used
// to point to an `string_data_item`

#ifndef KUNAI_DEX_PARSER_STRINGS_HPP
#define KUNAI_DEX_PARSER_STRINGS_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "Kunai/Utils/kunaistream.hpp"

namespace KUNAI
{
namespace DEX
{
    /// @brief container with all the created strings from the
    /// dex file, also it will be used to access by order.
    using ordered_strings_t = std::vector<std::string>;

    using offset_string_t = std::unordered_map<std::uint32_t, std::string*>;

    /// @brief Storage class for all the strings
    /// of the DEX file.
    class Strings
    {
        /// @brief offset of the table with the strings
        std::uint32_t strings_offset;
        /// @brief number of strings, returned for size of strings
        std::uint32_t number_of_strings;
        /// @brief variable with all the strings by id
        ordered_strings_t ordered_strings;
        /// @brief variable with all the strings by offset (pointer)
        offset_string_t offset_strings;

    public:
        /// @brief Constructor of string class
        Strings() = default;

        /// @brief Copy constructors of strings
        /// @param str another strings object
        Strings(Strings &str);

        /// @brief destructor of the strings.
        ~Strings() = default;

        /// @brief Parse the strings from the DEX file
        /// @param strings_offset offset where to read the strings
        /// @param number_of_strings number of strings to read
        /// @param stream stream with file
        void parse_strings(std::uint32_t strings_offset, 
            std::uint32_t number_of_strings, 
            stream::KunaiStream* stream);

        /// @brief Return the offset strings map as constant
        /// @return offset-strings map
        const offset_string_t& get_offset_strings() const
        {
            return offset_strings;
        }

        /// @brief Return a pointer to an string giving an offset
        /// @param offset offset where string is
        /// @return pointer to string in the offset
        std::string* get_string_from_offset(std::uint32_t offset);

        /// @brief Get reference to string by a given id
        /// @param id id commonly refers to position
        /// @return const reference to string
        const std::string& get_string_by_id(std::uint32_t id) const;
        
        /// @brief Get the number of the strings stored.
        /// @return uint32_t with number of strings
        std::uint32_t get_number_of_strings() const
        {
            return number_of_strings;
        }

        /// @brief pretty print the list of strings.
        /// @param os stream where to print to
        /// @param entry entry with the strings
        /// @return 
        friend std::ostream& operator<<(std::ostream& os, const Strings& entry);

        /// @brief Dump the Strings to an XML
        /// @param fos xml where to dump
        void to_xml(std::ofstream& fos);
    };
}
}

#endif // KUNAI_DEX_PARSER_DEXSTRINGS_HPP