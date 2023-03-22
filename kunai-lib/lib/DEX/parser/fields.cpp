//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file fields.cpp

#include "Kunai/DEX/parser/fields.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"

using namespace KUNAI::DEX;

std::string& FieldID::pretty_field()
{
    if (!pretty_name.empty())
        return pretty_name;

    pretty_name = class_->get_raw() + "->" + name_ + " " + type_->get_raw();
    return pretty_name;
}

void Fields::parse_fields(
    stream::KunaiStream *stream,
    Types *types,
    Strings *strings,
    std::uint32_t fields_offset,
    std::uint32_t fields_size)
{
    auto current_offset = stream->tellg();
    this->fields_size = fields_size;
    std::uint16_t class_idx, type_idx;
    std::uint32_t name_idx;
    fieldid_t fieldid;

    // move to the offset where ids are
    stream->seekg(fields_offset, std::ios_base::beg);

    for (size_t I = 0; I < fields_size; ++I)
    {
        stream->read_data<std::uint16_t>(class_idx, sizeof(std::uint16_t));

        stream->read_data<std::uint16_t>(type_idx, sizeof(std::uint16_t));

        stream->read_data<std::uint32_t>(name_idx, sizeof(std::uint32_t));
        // create the object with the information
        fieldid = std::make_unique<FieldID>(
            types->get_type_from_order(class_idx),
            types->get_type_from_order(type_idx),
            strings->get_string_by_id(name_idx));
        // move the ownership to the vector
        fields.push_back(std::move(fieldid));
    }

    // return to the previous offset
    stream->seekg(current_offset, std::ios_base::beg);
}

FieldID *Fields::get_field(std::uint32_t pos)
{
    if (pos >= fields.size())
        throw exceptions::IncorrectIDException("fields.cpp: position for field incorrect");
    return fields[pos].get();
}

std::ostream &operator<<(std::ostream &os, const Fields &entry)
{
    size_t I = 0;
    const auto &fields = entry.get_fields();

    os << "DEX Fields:\n";

    for (auto &field : fields)
    {
        os << "Field(" << I++ << "): " << field->pretty_field() << "\n";
    }

    return os;
}

void Fields::to_xml(std::ofstream &fos)
{
    fos << "<fields>\n";
    for (auto &field : fields)
    {
        fos << "\t<field>\n";
        fos << "\t\t<type>" << field->get_type()->get_raw() << "</type>\n";
        fos << "\t\t<name>" << field->get_name() << "</name>\n";
        fos << "\t\t<class>" << field->get_class()->get_raw() << "</type>\n";
        fos << "\t</field>\n";
    }
    fos << "</fields>\n";
}