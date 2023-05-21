//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file map_item.cpp

#include "Kunai/DEX/parser/map_item.hpp"
#include "Kunai/Utils/logger.hpp"

using namespace KUNAI::DEX;

void MapList::parse_map_list(stream::KunaiStream *stream,
                             std::uint32_t map_off)
{
    auto logger = LOGGER::logger();

    auto current_offset = stream->tellg();
    std::uint32_t size;

    map_item item;

    logger->debug("map_item.cpp: start parsing of map list");

    stream->seekg(map_off, std::ios_base::beg);

    // first read the size
    stream->read_data<std::uint32_t>(size, sizeof(std::uint32_t));

    for (size_t I = 0; I < size; ++I)
    {
        stream->read_data<map_item>(item, sizeof(map_item));

        items[item.type] = {item.type, item.unused, item.size, item.offset};
    }

    stream->seekg(current_offset, std::ios_base::beg);

    logger->debug("map_item.cpp: finished parsing of map list");
}

namespace KUNAI
{
namespace DEX
{
    std::ostream &operator<<(std::ostream &os, const MapList &entry)
    {
        size_t i = 0;
        const auto &map_items = entry.get_map_items();

        os << "DEX Map List:\n";
        for (auto &map_item : map_items){
            os << "Map item ("<< i++ <<"): ";
            auto type_name = entry.type_names.find(map_item.first);
            if (type_name != entry.type_names.end())
            {
                os << type_name->second << ", Size: " << map_item.second.size << ", Offset: " << map_item.second.offset << "\n"; 
            }
            else
            {
                os << "Unknown type code\n";
            }
        }
        return os;
    }

}
}