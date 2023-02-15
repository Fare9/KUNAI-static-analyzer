//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file methods.cpp

#include "Kunai/DEX/parser/methods.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"

using namespace KUNAI::DEX;


void Methods::parse_methods(
    stream::KunaiStream *stream,
    Types *types,
    Protos *protos,
    Strings *strings,
    std::uint32_t methods_offset,
    std::uint32_t methods_size)
{
    auto current_offset = stream->tellg();
    this->methods_size = methods_size;

    std::uint16_t class_idx;
    std::uint16_t proto_idx;
    std::uint32_t name_idx;
    methodid_t methodid;

    // move to the offset where ids are
    stream->seekg(methods_offset, std::ios_base::beg);

    for (size_t I = 0; I < methods_size; ++I)
    {
        stream->read_data<std::uint16_t>(class_idx, sizeof(std::uint16_t));

        stream->read_data<std::uint16_t>(proto_idx, sizeof(std::uint16_t));

        stream->read_data<std::uint32_t>(name_idx, sizeof(std::uint32_t));
        // create the object with the information
        methodid = std::make_unique<MethodID>(
            types->get_type_from_order(class_idx),
            protos->get_proto_by_order(proto_idx),
            strings->get_string_by_id(name_idx));
        // move the ownership to the vector
        methods.push_back(std::move(methodid));
    }

    // return to the previous offset
    stream->seekg(current_offset, std::ios_base::beg);
}

MethodID *Methods::get_method(std::uint32_t pos)
{
    if (pos >= methods.size())
        throw exceptions::IncorrectIDException("methods.cpp: position for method incorrect");
    return methods[pos].get();
}

std::ostream &operator<<(std::ostream &os, const Methods &entry)
{
    size_t I = 0;
    const auto &methods = entry.get_methods();

    os << "DEX Methods:\n";

    for (auto &method : methods)
    {
        os << "Method(" << I++ << "): " << method->pretty_method() << "\n";
    }

    return os;
}

void Methods::to_xml(std::ofstream &fos)
{
    fos << "<methods>\n";
    for (auto &method : methods)
    {
        fos << "\t<method>\n";
        fos << "\t\t<type>" << method->get_proto()->get_shorty_idx() << "</type>\n";
        fos << "\t\t<name>" << method->get_name() << "</name>\n";
        fos << "\t\t<class>" << method->get_class()->get_raw() << "</type>\n";
        fos << "\t</method>\n";
    }
    fos << "</methods>\n";
}