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
    for (auto val : str.offset_strings)
    {
        auto offset = val.first;
        auto ptr_str = val.second;

        ordered_strings.push_back(*ptr_str);

        offset_strings[offset] = &ordered_strings.back();
    }
}

void Strings::parse_strings(std::uint32_t strings_offset,
                            std::uint32_t number_of_strings,
                            stream::KunaiStream *stream)
{
    // utilities
    size_t I;
    auto logger = LOGGER::logger();
    auto current_offset = stream->tellg();
    // store it, will be useful...
    this->number_of_strings = number_of_strings;

    // values
    std::uint32_t str_offset;
    std::string str;

    // move to the offset where are the strings ids
    stream->seekg(strings_offset, std::ios_base::beg);

    logger->debug("started parsing strings");

    for (I = 0; I < number_of_strings; ++I)
    {
        stream->read_data<std::uint32_t>(str_offset, sizeof(std::uint32_t));

        if (str_offset > stream->get_size())
            throw exceptions::OutOfBoundException("strings.cpp: string offset out of bound");

        // read the strings from the offset and store it
        ordered_strings.push_back(stream->read_dex_string(str_offset));
        offset_strings[str_offset] = &ordered_strings.back();
    }

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

const std::string &Strings::get_string_by_id(std::uint32_t id) const
{
    if (id >= number_of_strings)
        throw exceptions::IncorrectIDException("strings.cpp: id for string incorrect");
    return ordered_strings[id];
}

std::ostream &operator<<(std::ostream &os, const Strings &entry)
{
    size_t I = 0;
    auto &offset_strings = entry.get_offset_strings();
    os << std::hex;
    os << std::setw(30) << std::left << std::setfill(' ') << "Dex Strings\n";
    for (const auto s : offset_strings)
        os << std::left << std::setfill(' ') << "String (" << std::dec << I++ << "): " << s.first << "->\"" << *s.second << "\"\n";
    return os;
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