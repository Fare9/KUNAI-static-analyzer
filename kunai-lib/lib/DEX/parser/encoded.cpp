//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file encoded.cpp

#include "Kunai/DEX/parser/encoded.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"

using namespace KUNAI::DEX;

void EncodedValue::parse_encoded_value(
    stream::KunaiStream *stream,
    Types *types,
    Strings *strings)
{
    std::uint8_t value;

    stream->read_data<std::uint8_t>(value, sizeof(std::uint8_t));

    value_type = static_cast<TYPES::value_format>(value & 0x1f);
    value_args = ((value & 0xe0) >> 5);

    switch (value_type)
    {
    case TYPES::value_format::VALUE_BYTE:
    case TYPES::value_format::VALUE_SHORT:
    case TYPES::value_format::VALUE_CHAR:
    case TYPES::value_format::VALUE_INT:
    case TYPES::value_format::VALUE_LONG:
    case TYPES::value_format::VALUE_FLOAT:
    case TYPES::value_format::VALUE_DOUBLE:
    case TYPES::value_format::VALUE_STRING:
    case TYPES::value_format::VALUE_TYPE:
    case TYPES::value_format::VALUE_FIELD:
    case TYPES::value_format::VALUE_METHOD:
    case TYPES::value_format::VALUE_ENUM:
    {
        std::uint8_t aux;
        for (size_t I = 0; I < value_args; ++I)
        {
            stream->read_data<std::uint8_t>(aux, sizeof(std::uint8_t));
            values.push_back(aux);
        }
        break;
    }
    case TYPES::value_format::VALUE_ARRAY:
        array_data.parse_encoded_array(stream, types, strings);
        break;
    case TYPES::value_format::VALUE_ANNOTATION:
        annotation.parse_encoded_annotation(stream, types, strings);
        break;
    default:
        break;
    }
}

void EncodedArray::parse_encoded_array(
    stream::KunaiStream *stream,
    Types *types,
    Strings *strings)
{
    array_size = stream->read_uleb128();
    encodedvalue_t encoded_value;

    for (size_t I = 0; I < array_size; ++I)
    {
        encoded_value = std::make_unique<EncodedValue>();
        encoded_value->parse_encoded_value(stream, types, strings);
        values.push_back(std::move(encoded_value));
    }
}

void EncodedAnnotation::parse_encoded_annotation(
    stream::KunaiStream *stream,
    Types *types,
    Strings *strings)
{
    annotationelement_t annotation_element;
    encodedvalue_t encoded_value;
    std::uint64_t name_idx;
    auto type_idx = stream->read_uleb128();
    size = stream->read_uleb128();

    type = types->get_type_from_order(static_cast<std::uint32_t>(type_idx));

    for (size_t I = 0; I < size; ++I)
    {
        // read first the name_idx, then the EncodedValue
        name_idx = stream->read_uleb128();
        // read the EncodedValue
        encoded_value = std::make_unique<EncodedValue>();
        encoded_value->parse_encoded_value(stream, types, strings);

        // create the AnnotationElement
        annotation_element = std::make_unique<AnnotationElement>(
            strings->get_string_by_id(name_idx),
            encoded_value);

        // push it into the elements vector
        elements.push_back(std::move(annotation_element));
    }
}

AnnotationElement *EncodedAnnotation::get_annotation_by_pos(std::uint32_t pos)
{
    if (pos >= elements.size())
        throw exceptions::IncorrectIDException("encoded.cpp: get_annotation_by_pos position given incorrect");
    return elements.at(pos).get();
}

void EncodedCatchHandler::parse_encoded_catch_handler(
    stream::KunaiStream *stream,
    Types *types)
{
    std::uint64_t type_idx, addr;
    encodedtypepair_t encoded_type_pair;

    size = stream->read_sleb128();

    for (size_t I = 0, S = std::abs(size); I < S; ++I)
    {
        type_idx = stream->read_uleb128();
        addr = stream->read_uleb128();

        encoded_type_pair = std::make_unique<EncodedTypePair>(
            types->get_type_from_order(type_idx),
            addr);
        handlers.push_back(std::move(encoded_type_pair));
    }

    // A size of 0 means that there is a catch-all but no explicitly typed catches
    // And a size of -1 means that there is one typed catch along with a catch-all.
    if (size <= 0)
        catch_all_addr = stream->read_uleb128();
}

EncodedTypePair *EncodedCatchHandler::get_handler_by_pos(std::uint64_t pos)
{
    if (pos >= handlers.size())
        throw exceptions::IncorrectIDException("encoded.cpp: EncodedTypePair value for position incorrect");
    return handlers.at(pos).get();
}

void TryItem::parse_try_item(stream::KunaiStream *stream)
{
    stream->read_data<try_item_struct_t>(try_item_struct, sizeof(try_item_struct_t));
}

void CodeItemStruct::parse_code_item_struct(
    stream::KunaiStream *stream,
    Types *types)
{
    // the instructions are read in chunks of 16 bits
    std::uint16_t instruction;
    size_t I;
    tryitem_t try_item;
    encodedcatchhandler_t encoded_catch_handler;

    // first we need to read the code_item_struct_t
    stream->read_data<code_item_struct_t>(code_item, sizeof(code_item_struct_t));

    // now we can work with the values

    // first read the instructions for the CodeItem
    for (I = 0; I < code_item.insns_size; ++I)
    {
        // read the instruction
        stream->read_data<std::uint16_t>(instruction, sizeof(std::uint16_t));

        instructions_raw.push_back(instruction & 0xFF);
        instructions_raw.push_back((instruction >> 8) & 0xFF);
    }

    if ((code_item.tries_size > 0) && // padding present in case tries_size > 0
        (code_item.insns_size % 2))   // and instructions size is odd
    {
        // padding advance 2 bytes
        stream->seekg(sizeof(std::uint16_t), std::ios_base::cur);
    }

    // check if there are try-catch stuff
    if (code_item.tries_size > 0)
    {
        for (I = 0; I < code_item.tries_size; ++I)
        {
            try_item = std::make_unique<TryItem>();
            try_item->parse_try_item(stream);
            try_items.push_back(std::move(try_item));
        }
        // now get the number of catch handlers
        encoded_catch_handler_size = stream->read_uleb128();

        for (I = 0; I < encoded_catch_handler_size; ++I)
        {
            encoded_catch_handler = std::make_unique<EncodedCatchHandler>();
            encoded_catch_handler->parse_encoded_catch_handler(stream, types);
            encoded_catch_handlers.push_back(std::move(encoded_catch_handler));
        }
    }
}

void EncodedMethod::parse_encoded_method(stream::KunaiStream *stream,
                                         std::uint64_t code_off,
                                         Types *types)
{
    auto current_offset = stream->tellg();

    if (code_off > 0)
    {
        stream->seekg(code_off, std::ios_base::beg);
        // parse the code item
        code_item.parse_code_item_struct(stream, types);
    }

    // return to current offset
    stream->seekg(current_offset, std::ios_base::beg);
}