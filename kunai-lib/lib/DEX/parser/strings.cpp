//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file strings.cpp

#include "Kunai/DEX/parser/strings.hpp"
#include "Kunai/Utils/logger.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"
#include "Kunai/Exceptions/outofbound_exception.hpp"

#include <iomanip>

using namespace KUNAI::DEX;

Strings::Strings(Strings &str)
{
    std::vector<std::uint32_t> str_offsets;

    for (auto val : str.offset_strings)
    {
        auto offset = val.first;
        auto ptr_str = val.second;

        str_offsets.push_back(offset);
        ordered_strings.push_back(*ptr_str);
    }

    for (size_t I = 0; I < ordered_strings.size(); ++I)
        offset_strings[str_offsets[I]] = &ordered_strings[I];
}

void Strings::parse_strings(std::uint32_t strings_offset,
                            std::uint32_t number_of_strings,
                            stream::KunaiStream *stream)
{
    std::vector<std::uint32_t> str_offsets;

    // utilities
    size_t I;
    auto logger = LOGGER::logger();
    auto current_offset = stream->tellg();
    // store it, will be useful...
    this->number_of_strings = number_of_strings;

    // values
    std::uint32_t str_offset;

    // move to the offset where are the strings ids
    stream->seekg(strings_offset, std::ios_base::beg);

    logger->debug("started parsing strings");

    for (I = 0; I < number_of_strings; ++I)
    {
        stream->read_data<std::uint32_t>(str_offset, sizeof(std::uint32_t));

        if (str_offset > stream->get_size())
            throw exceptions::OutOfBoundException("strings.cpp: string offset out of bound");

        // read the strings from the offset and store it
        str_offsets.push_back(str_offset); // keep the offset for later
        ordered_strings.push_back(stream->read_dex_string(str_offset));
    }
    /// store the offsets and the strings
    for (I = 0; I < ordered_strings.size(); ++I)
        offset_strings[str_offsets[I]] = &ordered_strings[I];

    // return to the stored offset
    stream->seekg(current_offset, std::ios_base::beg);
}

std::string *Strings::get_string_from_offset(std::uint32_t offset)
{
    auto it = offset_strings.find(offset);

    if (it == offset_strings.end())
        throw exceptions::IncorrectIDException("strings.cpp: offset for string incorrect");

    return it->second;
}

std::string &Strings::get_string_by_id(std::uint32_t id)
{
    if (id >= number_of_strings)
        throw exceptions::IncorrectIDException("strings.cpp: id for string incorrect");
    return ordered_strings[id];
}

void Strings::to_xml(std::ofstream &fos)
{
    fos << std::hex;
    fos << "<strings>\n";
    for (const auto &s : offset_strings)
    {
        fos << "\t<string>\n";
        fos << "\t\t<offset>" << s.first << "</offset>\n";
        fos << "\t\t<value>" << *s.second << "</value>\n";
        fos << "\t</string>\n";
    }
    fos << "</strings>\n";
}

namespace KUNAI
{
namespace DEX
{
    std::ostream &operator<<(std::ostream &os, const Strings &entry)
    {
        size_t I = 0;
        auto &ordered_strings = entry.get_ordered_strings();
        auto &offset_strings = entry.get_offset_strings();
        os << std::hex;
        os << "Dex Strings\n";
        for (const auto s : offset_strings)
            os << std::left << std::setfill(' ') << "String (" << std::dec << I++ << "): " << s.first << "->\"" << *s.second << "\"\n";
        return os;
    }
}
}