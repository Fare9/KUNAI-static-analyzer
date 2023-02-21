//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file annotations.hpp
// @brief Manage the annotations from the DEX file

#ifndef KUNAI_DEX_PARSER_ANNOTATIONS_HPP
#define KUNAI_DEX_PARSER_ANNOTATIONS_HPP

#include "Kunai/Utils/kunaistream.hpp"
#include "Kunai/DEX/parser/fields.hpp"
#include "Kunai/DEX/parser/methods.hpp"

#include <memory>
#include <vector>
#include <unordered_map>

namespace KUNAI
{
namespace DEX
{
    /// @brief Information for the list of annotations for the parameters
    class ParameterAnnotation
    {
        /// @brief idx of method for the annotations
        std::uint32_t method_idx;
        /// @brief offset where annotations are
        std::uint32_t annotations_off;
    public:
        /// @brief Constructor of ParameterAnnotation
        /// @param method_idx idx of the method
        /// @param annotations_off offset of the annotation
        ParameterAnnotation(std::uint32_t method_idx, std::uint32_t annotations_off)
            : method_idx(method_idx), annotations_off(annotations_off)
        {}

        /// @brief Destructor of ParameterAnnotation
        ~ParameterAnnotation() = default;

        /// @brief Get the MethodIDX value
        /// @return method idx value
        std::uint32_t get_method_idx() const
        {
            return method_idx;
        }

        /// @brief Get the offset to the annotations
        /// @return annotations offset
        std::uint32_t get_annotations_off() const
        {
            return annotations_off;
        }
    };

    using parameterannotation_t = std::unique_ptr<ParameterAnnotation>;

    /// @brief Information for the list of annotations for the methods
    class MethodAnnotation
    {
        /// @brief idx of the associated method
        std::uint32_t method_idx;
        /// @brief Offset to the annotations of the method
        std::uint32_t annotations_off;
    public:
        /// @brief Constructor of MethodAnnotations
        /// @param method_idx idx of the method of the annotations
        /// @param annotations_off offset with the annotations
        MethodAnnotation(std::uint32_t method_idx, std::uint32_t annotations_off)
            : method_idx(method_idx), annotations_off(annotations_off)
        {}
        /// @brief Destructor of the method annotations
        ~MethodAnnotation() = default;

        /// @brief Obtain the method idx of the current method annotation
        /// @return method idx of the annotation
        std::uint32_t get_method_idx() const
        {
            return method_idx;
        }
        /// @brief Obtain the offset of the annotations 
        /// @return offset of method annotations
        std::uint32_t get_annotations_off() const
        {
            return annotations_off;
        }
    };

    using methodannotation_t = std::unique_ptr<MethodAnnotation>;

    /// @brief Information for the list of annotations for the fields
    class FieldAnnotation
    {
        /// @brief Field IDX of the annotation
        std::uint32_t field_idx;
        /// @brief Offset to the annotations
        std::uint32_t annotations_off;
    public:

        /// @brief Constructor of FieldAnnotation
        /// @param field_idx idx of the field
        /// @param annotations_off offset to the annotations
        FieldAnnotation(std::uint32_t field_idx, std::uint32_t annotations_off)
            : field_idx(field_idx), annotations_off(annotations_off)
        {}

        /// @brief Destructor of FieldAnnotation
        ~FieldAnnotation() = default;

        /// @brief Get the field idx of the annotation
        /// @return field idx of the annotation
        std::uint32_t get_field_idx() const
        {
            return field_idx;
        }

        /// @brief Get the offset of the annotation
        /// @return offset of the annotation
        std::uint32_t get_annotations_off() const
        {
            return annotations_off;
        }
    };

    using fieldannotation_t = std::unique_ptr<FieldAnnotation>;

    /// @brief Class with all the previos annotations
    class AnnotationDirectoryItem
    {
        /// @brief Offset to the annotations of the class
        std::uint32_t class_annotations_off;
        /// @brief list of the field annotations
        std::vector<fieldannotation_t> field_annotations;
        /// @brief field annotations by id
        std::unordered_map<std::uint32_t, FieldAnnotation*> field_annotations_by_id;
        /// @brief list of method annotations
        std::vector<methodannotation_t> method_annotations;
        /// @brief method annotations by id
        std::unordered_map<std::uint32_t, MethodAnnotation*> method_annotations_by_id;
        /// @brief list of parameter annotations
        std::vector<parameterannotation_t> parameter_annotations;
        /// @brief parameter annotations by id
        std::unordered_map<std::uint32_t, ParameterAnnotation*> parameter_annotations_by_id;
    public:
        /// @brief Constructor of AnnotationDirectoryItem
        AnnotationDirectoryItem() = default;
        /// @brief Destructor of AnnotationDirectoryItem
        ~AnnotationDirectoryItem() = default;
        /// @brief Parse the annotation directory item
        /// @param stream stream with the DEX file
        void parse_annotation_directory_item(stream::KunaiStream* stream);

        /// @brief Get a constant reference to all the
        /// field annotations from the annotation directory
        /// @return constant reference to vector of field annotations
        const std::vector<fieldannotation_t>& get_field_annotations() const
        {
            return field_annotations;
        }

        /// @brief Get a reference to all the field
        /// annotations from the annotation directory
        /// @return reference to vector of field annotations
        std::vector<fieldannotation_t>& get_field_annotations()
        {
            return field_annotations;
        }

        /// @brief Get a pointer to a field annotation
        /// @param idx idx of the Field
        /// @return field annotation pointer
        FieldAnnotation* get_field_annotation_by_id(std::uint32_t idx);

        /// @brief Get a constant reference to all method annotations
        /// from the annotation directory
        /// @return constant reference to all method annotations
        const std::vector<methodannotation_t>& get_method_annotations() const
        {
            return method_annotations;
        }

        /// @brief Get a reference to all method annotations
        /// from the annotation directory
        /// @return reference to all method annotations
        std::vector<methodannotation_t>& get_method_annotations()
        {
            return method_annotations;
        }

        /// @brief Get a pointer to method annotation
        /// @param idx idx of the Method
        /// @return method annotation pointer
        MethodAnnotation* get_method_annotation_by_id(std::uint32_t idx);

        /// @brief Get a constant reference to all the parameter annotations
        /// from the annotation directory
        /// @return constant reference to parameter annotations
        const std::vector<parameterannotation_t>& get_parameter_annotations() const
        {
            return parameter_annotations;
        }

        /// @brief Get a reference to all the parameter annotations
        /// from the annotation directory
        /// @return constant to parameter annotations
        std::vector<parameterannotation_t>& get_parameter_annotations()
        {
            return parameter_annotations;
        }

        /// @brief Get a pointer to a parameter annotation 
        /// @param idx idx of the method
        /// @return pointer to parameter annotations of the method
        ParameterAnnotation* get_parameter_annotation_by_id(std::uint32_t idx);
    };
} // namespace DEX
} // namespace KUNAI


#endif