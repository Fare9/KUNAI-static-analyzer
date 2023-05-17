//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file annotations.cpp

#include "Kunai/DEX/parser/annotations.hpp"
#include "Kunai/Exceptions/incorrectid_exception.hpp"

using namespace KUNAI::DEX;

void AnnotationDirectoryItem::parse_annotation_directory_item(stream::KunaiStream* stream)
{
    auto current_offset = stream->tellg();

    size_t I;
    std::uint32_t fields_size;
    std::uint32_t annotated_methods_size;
    std::uint32_t annotated_parameters_size;
    std::uint32_t idx;
    std::uint32_t annotations_off;

    fieldannotation_t fieldannotation;
    methodannotation_t methodannotation;
    parameterannotation_t parameterannotation;

    // read first an class annotations offset
    stream->read_data<std::uint32_t>(class_annotations_off, sizeof(std::uint32_t));

    stream->read_data<std::uint32_t>(fields_size, sizeof(std::uint32_t));

    stream->read_data<std::uint32_t>(annotated_methods_size, sizeof(std::uint32_t));

    stream->read_data<std::uint32_t>(annotated_parameters_size, sizeof(std::uint32_t));

    for (I = 0; I < fields_size; ++I)
    {
        stream->read_data<std::uint32_t>(idx, sizeof(std::uint32_t));
        stream->read_data<std::uint32_t>(annotations_off, sizeof(std::uint32_t));
        fieldannotation = std::make_unique<FieldAnnotation>(idx, annotations_off);
        field_annotations.push_back(std::move(fieldannotation));
        field_annotations_by_id[idx] = field_annotations.back().get();
    }

    for (I = 0; I < annotated_methods_size; ++I)
    {
        stream->read_data<std::uint32_t>(idx, sizeof(std::uint32_t));
        stream->read_data<std::uint32_t>(annotations_off, sizeof(std::uint32_t));
        methodannotation = std::make_unique<MethodAnnotation>(idx, annotations_off);
        method_annotations.push_back(std::move(methodannotation));
        method_annotations_by_id[idx] = method_annotations.back().get();
    }

    for (I = 0; I < annotated_parameters_size; ++I)
    {
        stream->read_data<std::uint32_t>(idx, sizeof(std::uint32_t));
        stream->read_data<std::uint32_t>(annotations_off, sizeof(std::uint32_t));
        parameterannotation = std::make_unique<ParameterAnnotation>(idx, annotations_off);
        parameter_annotations.push_back(std::move(parameterannotation));
    }

    stream->seekg(current_offset, std::ios_base::beg);
}

FieldAnnotation* AnnotationDirectoryItem::get_field_annotation_by_id(std::uint32_t idx)
{
    auto it = field_annotations_by_id.find(idx);

    if (it == field_annotations_by_id.end())
        throw exceptions::IncorrectIDException("get_field_annotation_by_id(): idx provided incorrect");
    
    return it->second;
}

MethodAnnotation* AnnotationDirectoryItem::get_method_annotation_by_id(std::uint32_t idx)
{
    auto it = method_annotations_by_id.find(idx);
    
    if (it == method_annotations_by_id.end())
        throw exceptions::IncorrectIDException("get_method_annotation_by_id(): idx provided incorrect");
    
    return it->second;
}

ParameterAnnotation* AnnotationDirectoryItem::get_parameter_annotation_by_id(std::uint32_t idx)
{
    auto it = parameter_annotations_by_id.find(idx);

    if (it == parameter_annotations_by_id.end())
        throw exceptions::IncorrectIDException("get_parameter_annotation_by_id(): idx provided incorrect");
    
    return it->second;
}