//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file parser.hpp
// @brief class representing the DEX parser it contains all the headers
// from the DEX file.
#ifndef KUNAI_DEX_PARSER_PARSER_HPP
#define KUNAI_DEX_PARSER_PARSER_HPP

#include "Kunai/DEX/parser/header.hpp"
#include "Kunai/DEX/parser/strings.hpp"
#include "Kunai/DEX/parser/types.hpp"
#include "Kunai/Utils/kunaistream.hpp"


namespace KUNAI
{
namespace DEX
{

    class Parser
    {
        /// @brief Dex header class
        /// with the dex header structure
        Header header;

        /// @brief Dex strings used
        /// during the whole dex structures
        Strings strings;

        /// @brief Types of the DEX file
        Types types;

        /// @brief stream with the file
        stream::KunaiStream* stream;

    public:

        /// @brief Constructor of the parser
        /// @param stream stream where to read the data
        Parser(stream::KunaiStream* stream) : stream(stream)
        {}

        /// @brief Destructor of the parser
        ~Parser() = default;

        /// @brief parse the dex file and obtain the different objects
        void parse_file();

        /// @brief Return a const reference from the dex header
        /// @return const dex header reference
        const Header& get_header_const() const
        {
            return header;
        }

        /// @brief Return a reference from the dex header
        /// @return dex header reference
        Header& get_header()
        {
            return header;
        }

        /// @brief get a reference to the strings object, reference is constant
        /// @return constant reference to strings
        const Strings& get_strings_const() const
        {
            return strings;
        }

        /// @brief get a reference to the strings object
        /// @return reference to strings
        Strings& get_strings()
        {
            return strings;
        }

        /// @brief get a constant reference to the types
        /// @return constant reference to types
        const Types& get_types_const() const
        {
            return types;
        }

        /// @brief get a reference to the types object
        /// @return reference to types
        Types& get_types()
        {
            return types;
        }
    };
} // namespace DEX
} // namespace KUNAI


#endif