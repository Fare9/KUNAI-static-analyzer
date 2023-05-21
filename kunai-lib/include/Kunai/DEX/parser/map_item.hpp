//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file map_item.hpp
// The map contains all the different data from the DEX file contained in
// the data section. Androguard retrieves all the information from here, but
// in some cases, contains less data than the header.
#ifndef KUNAI_DEX_PARSER_MAP_ITEM_HPP
#define KUNAI_DEX_PARSER_MAP_ITEM_HPP

#include "Kunai/Utils/kunaistream.hpp"
#include <unordered_map>

namespace KUNAI
{
namespace DEX
{
    /// @brief Class representing the Map List, this map contains
    /// different information about different types
    class MapList
    {
    public:
        /// @brief all possible type codes
        enum type_codes : std::uint16_t
        {
            TYPE_HEADER_ITEM = 0x0000,
            TYPE_STRING_ID_ITEM = 0x0001,
            TYPE_TYPE_ID_ITEM = 0x0002,
            TYPE_PROTO_ID_ITEM = 0x0003,
            TYPE_FIELD_ID_ITEM = 0x0004,
            TYPE_METHOD_ID_ITEM = 0x0005,
            TYPE_CLASS_DEF_ITEM = 0x0006,
            TYPE_CALL_SITE_ID_ITEM = 0x0007,
            TYPE_METHOD_HANDLE_ITEM = 0x0008,
            TYPE_MAP_LIST = 0x1000,
            TYPE_TYPE_LIST = 0x1001,
            TYPE_ANNOTATION_SET_REF_LIST = 0x1002,
            TYPE_ANNOTATION_SET_ITEM = 0x1003,
            TYPE_CLASS_DATA_ITEM = 0x2000,
            TYPE_CODE_ITEM = 0x2001,
            TYPE_STRING_DATA_ITEM = 0x2002,
            TYPE_DEBUG_INFO_ITEM = 0x2003,
            TYPE_ANNOTATION_ITEM = 0x2004,
            TYPE_ENCODED_ARRAY_ITEM = 0x2005,
            TYPE_ANNOTATIONS_DIRECTORY_ITEM = 0x2006,
            TYPE_HIDDENAPI_CLASS_DATA_ITEM = 0xF000
        };

        /// @brief Map to store each item type name with its value
        std::unordered_map<std::uint16_t, std::string> type_names = {
            {TYPE_HEADER_ITEM, "TYPE_HEADER_ITEM"},
            {TYPE_STRING_ID_ITEM, "TYPE_STRING_ID_ITEM"},
            {TYPE_TYPE_ID_ITEM, "TYPE_TYPE_ID_ITEM"},
            {TYPE_PROTO_ID_ITEM, "TYPE_PROTO_ID_ITEM"},
            {TYPE_FIELD_ID_ITEM, "TYPE_FIELD_ID_ITEM"},
            {TYPE_METHOD_ID_ITEM, "TYPE_METHOD_ID_ITEM"},
            {TYPE_CLASS_DEF_ITEM, "TYPE_CLASS_DEF_ITEM"},
            {TYPE_CALL_SITE_ID_ITEM, "TYPE_CALL_SITE_ID_ITEM"},
            {TYPE_METHOD_HANDLE_ITEM, "TYPE_METHOD_HANDLE_ITEM"},
            {TYPE_MAP_LIST, "TYPE_MAP_LIST"},
            {TYPE_TYPE_LIST, "TYPE_TYPE_LIST"},
            {TYPE_ANNOTATION_SET_REF_LIST, "TYPE_ANNOTATION_SET_REF_LIST"},
            {TYPE_ANNOTATION_SET_ITEM, "TYPE_ANNOTATION_SET_ITEM"},
            {TYPE_CLASS_DATA_ITEM, "TYPE_CLASS_DATA_ITEM"},
            {TYPE_CODE_ITEM, "TYPE_CODE_ITEM"},
            {TYPE_STRING_DATA_ITEM, "TYPE_STRING_DATA_ITEM"},
            {TYPE_DEBUG_INFO_ITEM, "TYPE_DEBUG_INFO_ITEM"},
            {TYPE_ANNOTATION_ITEM, "TYPE_ANNOTATION_ITEM"},
            {TYPE_ENCODED_ARRAY_ITEM, "TYPE_ENCODED_ARRAY_ITEM"},
            {TYPE_ANNOTATIONS_DIRECTORY_ITEM, "TYPE_ANNOTATIONS_DIRECTORY_ITEM"},
            {TYPE_HIDDENAPI_CLASS_DATA_ITEM, "TYPE_HIDDENAPI_CLASS_DATA_ITEM"}
        };

        /// @brief Map that store the information of the map
        struct map_item
        {
            type_codes type;            //! type of the item
            std::uint16_t unused;       //! not used, do not retrieve it
            std::uint32_t size;         //! number of items to be found on the offset
            std::uint32_t offset;       //! offset where to read the items
        };
        
    private:
        /// @brief map of items, each type code will contain a map item
        std::unordered_map<type_codes, map_item> items;
    public:
        /// @brief Constructor of the MapList
        MapList() = default;
        /// @brief Destructor of the MapList
        ~MapList() = default;

        /// @brief Parse the map list from the DEX file to create the map
        /// @param stream DEX file content
        /// @param map_off offset to the map_list
        void parse_map_list(stream::KunaiStream* stream, std::uint32_t map_off);

        /// @brief Get the map items from the DEX 
        /// @return constant reference to the map items
        const std::unordered_map<type_codes, map_item>& get_map_items() const
        {
            return items;
        }

        /// @brief Get the map items from the DEX
        /// @return reference to the map items
        std::unordered_map<type_codes, map_item>& get_map_items()
        {
            return items;
        }

        /// @brief Pretty printer for the operator << of the DEX map list
        /// @param os stream to print the DEX map list
        /// @param entry entry to print
        /// @return stream  
        friend std::ostream& operator<<(std::ostream& os, const MapList& entry);
    };
} // namespace DEX
} // namespace KUNAI


#endif