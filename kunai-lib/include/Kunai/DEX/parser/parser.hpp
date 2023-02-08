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
#include "Kunai/Utils/kunaistream.hpp"

namespace KUNAI
{
namespace DEX
{

    class Parser
    {
        /// @brief Dex header class
        /// with the dex header structure
        std::unique_ptr<Header> header;

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

        /// @brief Return a const pointer from the dex header
        /// @return const dex header pointer
        const Header* get_header_const() const
        {
            return header.get();
        }

        /// @brief Return a pointer from the dex header
        /// @return dex header pointer
        Header* get_header()
        {
            return header.get();
        }
    };
} // namespace DEX
} // namespace KUNAI


#endif