//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file classes.cpp

#include "Kunai/Utils/logger.hpp"
#include "Kunai/DEX/parser/classes.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"

using namespace KUNAI::DEX;

void ClassDataItem::parse_class_data_item(
    stream::KunaiStream *stream,
    Fields *fields,
    Methods *methods,
    Types *types)
{
    auto current_offset = stream->tellg();
    std::uint64_t I;
    // IDs for the different variables
    std::uint64_t static_field = 0,
                  instance_field = 0,
                  direct_method = 0,
                  virtual_method = 0;
    std::uint64_t access_flags; // access flags of the variables
    std::uint64_t code_offset;  // offset for parsing

    // read the sizes of the different variables
    std::uint64_t static_fields_size = stream->read_uleb128();
    std::uint64_t instance_fields_size = stream->read_uleb128();
    std::uint64_t direct_methods_size = stream->read_uleb128();
    std::uint64_t virtual_methods_size = stream->read_uleb128();

    for (I = 0; I < static_fields_size; ++I)
    {
        //! this value needs to be incremented
        //! with the uleb128 read, so we have
        //! static_field = prev + read
        static_field += stream->read_uleb128();

        //! read access flags value
        access_flags = stream->read_uleb128();

        // create the static field and the entry of the map
        static_fields.push_back(
            std::make_unique<EncodedField>(fields->get_field(static_field),
                                           static_cast<TYPES::access_flags>(access_flags)));
        static_fields_by_id[static_field] = static_fields.back().get();
    }

    for (I = 0; I < instance_fields_size; ++I)
    {
        instance_field += stream->read_uleb128();

        access_flags = stream->read_uleb128();

        instance_fields.push_back(
            std::make_unique<EncodedField>(fields->get_field(instance_field),
                                           static_cast<TYPES::access_flags>(access_flags)));
        instance_fields_by_id[instance_field] = instance_fields.back().get();
    }

    for (I = 0; I < direct_methods_size; ++I)
    {
        direct_method += stream->read_uleb128();

        access_flags = stream->read_uleb128();

        // for the code item
        code_offset = stream->read_uleb128();

        direct_methods.push_back(
            std::make_unique<EncodedMethod>(methods->get_method(direct_method),
                                            static_cast<TYPES::access_flags>(access_flags)));
        direct_methods_by_id[direct_method] = direct_methods.back().get();
        direct_methods.back()->parse_encoded_method(stream, code_offset, types);
    }

    for (I = 0; I < virtual_methods_size; ++I)
    {
        virtual_method += stream->read_uleb128();

        access_flags = stream->read_uleb128();

        code_offset = stream->read_uleb128();

        virtual_methods.push_back(
            std::make_unique<EncodedMethod>(methods->get_method(virtual_method),
                                            static_cast<TYPES::access_flags>(access_flags)));
        virtual_methods_by_id[virtual_method] = virtual_methods.back().get();
        virtual_methods.back()->parse_encoded_method(stream, code_offset, types);
    }

    stream->seekg(current_offset, std::ios_base::beg);
}

EncodedField *ClassDataItem::get_static_field_by_order(std::uint32_t ord)
{
    if (ord >= static_fields.size())
        throw exceptions::IncorrectIDException("get_static_field_by_order(): ord value given incorrect");
    return static_fields.at(ord).get();
}

EncodedField *ClassDataItem::get_static_field_by_id(std::uint32_t id)
{
    auto it = static_fields_by_id.find(id);

    if (it == static_fields_by_id.end())
        throw exceptions::IncorrectIDException("get_static_field_by_id(): id value given incorrect");
    return it->second;
}

EncodedField *ClassDataItem::get_instance_field_by_order(std::uint32_t ord)
{
    if (ord >= instance_fields.size())
        throw exceptions::IncorrectIDException("get_instance_field_by_order(): ord value given incorrect");
    return instance_fields.at(ord).get();
}

EncodedField *ClassDataItem::get_instance_field_by_id(std::uint32_t id)
{
    auto it = instance_fields_by_id.find(id);

    if (it == instance_fields_by_id.end())
        throw exceptions::IncorrectIDException("get_instance_field_by_id(): id value given incorrect");
    return it->second;
}

EncodedMethod *ClassDataItem::get_direct_method_by_order(std::uint32_t ord)
{
    if (ord >= direct_methods.size())
        throw exceptions::IncorrectIDException("get_direct_method_by_order(): ord value given incorrect");
    return direct_methods.at(ord).get();
}

EncodedMethod *ClassDataItem::get_direct_method_by_id(std::uint32_t id)
{
    auto it = direct_methods_by_id.find(id);

    if (it == direct_methods_by_id.end())
        throw exceptions::IncorrectIDException("get_direct_method_by_id(): id value given incorrect");
    return it->second;
}

EncodedMethod *ClassDataItem::get_virtual_method_by_order(std::uint32_t ord)
{
    if (ord >= virtual_methods.size())
        throw exceptions::IncorrectIDException("get_virtual_method_by_order(): ord value given incorrect");
    return virtual_methods.at(ord).get();
}

EncodedMethod *ClassDataItem::get_virtual_method_by_id(std::uint32_t id)
{
    auto it = virtual_methods_by_id.find(id);

    if (it == virtual_methods_by_id.end())
        throw exceptions::IncorrectIDException("get_virtual_method_by_id(): id value given incorrect");
    return it->second;
}

std::vector<EncodedField *> &ClassDataItem::get_fields()
{
    if (fields.size() == 0)
    {
        for (auto &field : static_fields)
            fields.push_back(field.get());
        for (auto &field : instance_fields)
            fields.push_back(field.get());
    }

    return fields;
}

std::vector<EncodedMethod *> &ClassDataItem::get_methods()
{
    if (methods.size() == 0)
    {
        for (auto &method : direct_methods)
            methods.push_back(method.get());
        for (auto &method : virtual_methods)
            methods.push_back(method.get());
    }

    return methods;
}

void ClassDef::parse_class_def(stream::KunaiStream *stream,
                               Strings *strings,
                               Types *types,
                               Fields *fields,
                               Methods *methods)
{
    auto current_offset = stream->tellg();

    size_t I;
    std::uint32_t size;
    std::uint16_t idx;

    
    // first of all read the classdefstruct_t
    stream->read_data<classdefstruct_t>(classdefstruct, sizeof(classdefstruct_t));

    // assign the class idx
    class_idx = reinterpret_cast<DVMClass *>(types->get_type_from_order(classdefstruct.class_idx));

    // assign the super class
    if (classdefstruct.superclass_idx != DEX::NO_INDEX)
        superclass_idx = reinterpret_cast<DVMClass *>(types->get_type_from_order(classdefstruct.superclass_idx));

    // assign the source file
    if (classdefstruct.source_file_idx != DEX::NO_INDEX)
        source_file = strings->get_string_by_id(classdefstruct.source_file_idx);

    // we can start now the parsing of the rest of the file!
    if (classdefstruct.interfaces_off)
    {
        stream->seekg(classdefstruct.interfaces_off, std::ios_base::beg);

        stream->read_data<std::uint32_t>(size, sizeof(std::uint32_t));

        for (I = 0; I < size; ++I)
        {
            stream->read_data<std::uint16_t>(idx, sizeof(std::uint16_t));

            interfaces.push_back(
                reinterpret_cast<DVMClass *>(types->get_type_from_order(idx)));
        }
    }

    if (classdefstruct.annotations_off)
    {
        // Implement parsing of annotations
    }

    if (classdefstruct.class_data_off)
    {
        stream->seekg(classdefstruct.class_data_off, std::ios_base::beg);

        class_data_item.parse_class_data_item(stream, fields, methods, types);
    }

    if (classdefstruct.static_values_off)
    {
        stream->seekg(classdefstruct.static_values_off, std::ios_base::beg);

        // the number of static values, is the number of static
        // fields of the class
        size = class_data_item.get_number_of_static_fields();

        encodedarray_t encodedarray;

        for (I = 0; I < size; ++I)
        {
            encodedarray = std::make_unique<EncodedArray>();
            encodedarray->parse_encoded_array(stream, types, strings);
            static_values.push_back(std::move(encodedarray));
        }
    }

    stream->seekg(current_offset, std::ios_base::beg);
}

void Classes::parse_classes(
    stream::KunaiStream *stream,
    std::uint32_t number_of_classes,
    std::uint32_t offset,
    Strings *strings,
    Types *types,
    Fields *fields,
    Methods *methods)
{
    auto logger = LOGGER::logger();
    auto current_offset = stream->tellg();
    this->number_of_classes = number_of_classes;

    classdef_t classdef;
    size_t I;

    logger->debug("classes.cpp: started parsing classes");

    // move to the offset
    stream->seekg(offset, std::ios_base::beg);

    for (I = 0; I < number_of_classes; ++I)
    {
        classdef = std::make_unique<ClassDef>();
        classdef->parse_class_def(stream, strings, types, fields, methods);
        class_defs.push_back(std::move(classdef));
        // since classdef restore the pointer it found, move it to next
        // structure
        stream->seekg(sizeof(ClassDef::classdefstruct_t), std::ios_base::cur);
    }

    logger->debug("classes.cpp: finished parsing classes");

    stream->seekg(current_offset, std::ios_base::beg);
}