//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file types.cpp
#include "Kunai/DEX/parser/types.hpp"
#include "Kunai/Utils/logger.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"

#include <iomanip>

using namespace KUNAI::DEX;

DVMType *Types::get_type_by_id(std::uint32_t type_id)
{
    auto it = types_by_id.find(type_id);

    if (it == types_by_id.end())
        throw exceptions::IncorrectIDException("types.cpp: id for type not found");

    return it->second;
}

DVMType *Types::get_type_from_order(std::uint32_t pos)
{
    if (pos >= ordered_types.size())
        throw exceptions::IncorrectIDException("types.cpp: position for type incorrect");

    return ordered_types[pos].get();
}

const std::string& DVMClass::pretty_print()
{
    if (!pretty_name.empty())
        return pretty_name;

    pretty_name = name.substr(1, name.size() - 2);
    std::replace(std::begin(pretty_name), std::end(pretty_name), '/', '.');
    return pretty_name;
}

const std::string& DVMArray::pretty_print()
{
    if (!pretty_name.empty())
        return pretty_name;
    pretty_name = array_type->pretty_print();
    for (size_t I = 0; I < depth; ++I)
        pretty_name += "[]";
    return pretty_name;
}

dvmtype_t Types::parse_type(std::string &name)
{
    switch (name.at(0))
    {

    case 'Z':
        return std::make_unique<DVMFundamental>(DVMFundamental::BOOLEAN, name);
    case 'B':
        return std::make_unique<DVMFundamental>(DVMFundamental::BYTE, name);
    case 'C':
        return std::make_unique<DVMFundamental>(DVMFundamental::CHAR, name);
    case 'D':
        return std::make_unique<DVMFundamental>(DVMFundamental::DOUBLE, name);
    case 'F':
        return std::make_unique<DVMFundamental>(DVMFundamental::FLOAT, name);
    case 'I':
        return std::make_unique<DVMFundamental>(DVMFundamental::INT, name);
    case 'J':
        return std::make_unique<DVMFundamental>(DVMFundamental::LONG, name);
    case 'S':
        return std::make_unique<DVMFundamental>(DVMFundamental::SHORT, name);
    case 'V':
        return std::make_unique<DVMFundamental>(DVMFundamental::VOID, name);
    case 'L':
        return std::make_unique<DVMClass>(name);
    case '[':
    {
        size_t depth = 0;
        const char *name_cpy = name.c_str();
        while (name_cpy[0] == '[')
        {
            depth++;
            name_cpy++;
        }
        std::string aux(name_cpy);
        dvmtype_t aux_type = parse_type(aux);
        return std::make_unique<DVMArray>(name, depth, aux_type);
    }
    default:
        return std::make_unique<Unknown>(name);
    }
}

void Types::parse_types(
    stream::KunaiStream *stream,
    Strings *strings,
    std::uint32_t number_of_types,
    std::uint32_t types_offset)
{
    auto logger = LOGGER::logger();
    auto current_offset = stream->tellg();
    this->number_of_types = number_of_types;
    this->offset = types_offset;
   
    dvmtype_t type;
    std::uint32_t type_id;

    logger->debug("started parsing types");

    // move to the offset for the analysis
    stream->seekg(types_offset, std::ios_base::beg);

    for (size_t I = 0; I < number_of_types; ++I)
    {
        stream->read_data<std::uint32_t>(type_id, sizeof(std::uint32_t));

        type = parse_type(strings->get_string_by_id(type_id));

        ordered_types.push_back(std::move(type));
        types_by_id[type_id] = ordered_types.back().get();
    }

    // return to your position
    stream->seekg(current_offset, std::ios_base::beg);

    logger->debug("finished parsing types");
}

std::ostream &operator<<(std::ostream &os, const Types &entry)
{
    const auto & types = entry.get_ordered_types();

    os << std::setw(30) << std::left << std::setfill(' ') << "DEX Types:\n";

    for (size_t I = 0; I < types.size(); ++I)
        os << std::left << std::setfill(' ') << "Type (" << I << ") -> \"" << types[I]->pretty_print() << "\"\n";
    
    return os;
}

void Types::to_xml(std::ofstream& xml_file)
{
    xml_file << "<types>\n";

    for (size_t I = 0; I < ordered_types.size(); ++I)
    {
        xml_file << "\t<type>\n";
        xml_file << "\t\t<id>" << I << "</id>\n";
        xml_file << "\t\t<value>" << ordered_types[I]->get_raw() << "</value>\n";
        xml_file << "\t</type>\n";
    }

    xml_file << "</types>\n";
}