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
#include "Kunai/DEX/parser/map_item.hpp"
#include "Kunai/DEX/parser/strings.hpp"
#include "Kunai/DEX/parser/types.hpp"
#include "Kunai/DEX/parser/protos.hpp"
#include "Kunai/DEX/parser/fields.hpp"
#include "Kunai/DEX/parser/methods.hpp"
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

        /// @brief Dex map list with information
        /// about the DEX, it looks different to
        /// the one of the header
        MapList maplist;

        /// @brief Dex strings used
        /// during the whole dex structures
        Strings strings;

        /// @brief Types of the DEX file
        Types types;

        /// @brief Protos of the methods from DEX file
        Protos protos;

        /// @brief Fields from all the DEX file
        Fields fields;

        /// @brief Methods from the DEX file
        Methods methods;

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

        /// @brief Return a constant reference to the maplist with the map items
        /// @return constant reference to maplist
        const MapList& get_maplist_const() const
        {
            return maplist;
        }

        /// @brief Return a reference to the maplist with the map items
        /// @return reference to maplist
        MapList& get_maplist()
        {
            return maplist;
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

        /// @brief get a constant reference to the prototypes
        /// @return constant reference to protos
        const Protos& get_protos_const() const
        {
            return protos;
        }

        /// @brief get a reference to the prototypes
        /// @return reference to protos
        Protos& get_protos()
        {
            return protos;
        }

        /// @brief get a reference to the fields
        /// @return reference to fields
        Fields& get_fields()
        {
            return fields;
        }

        /// @brief get a reference to the methods
        /// @return reference to methods
        Methods& get_methods()
        {
            return methods;
        }
    };
} // namespace DEX
} // namespace KUNAI


#endif