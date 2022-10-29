/***
 * @file dex_annotations.hpp
 * @author @Farenain
 *
 * @brief Android classes used to represent
 *        annotations. These are pointed by
 *        the offset annotations_off from ClassDefStruct.
 *        Annotations_off points to:
 *
 *        AnnotationsDirectoryItem {
 *         uint class_annotations_off, // offset to annotations made directly on the class.
 *         uint fields_size, // number of fields annotation by this item.
 *         uint annotated_methods_size, // number of methods annotated by this item.
 *         uint annotated_parameters_size, // number of method parameters list annotated by this item.
 *         field_annotation[fields_size] field_annotations, // associated field annotations. Must be field_idx sorted in increasing order.
 *         method_annotation[methods_size] method_annotations, // associated method annotations. Must be method_idx sorted in increasing order
 *         parameter_annotation[parameters_size] parameter_annotations //  associated method parameter annotations. Must be method_idx sorted in increasing order.
 *        }
 *
 *        FieldAnnotation {
 *         uint field_idx, // index into field_ids
 *         uint annotations_off // offset of the file to the list of annotations for the field. Format specified by annotation_set_item.
 *        }
 *
 *        MethodAnnotation {
 *         uint method_idx, // index into method_ids of the method being annotated.
 *         uint annotations_off, // offset of the file to list of annotations for the method. Format specified by annotation_set_item.
 *        }
 *
 *        ParameterAnnotation {
 *         uint method_idx, // index into method_idx of the method whose parameters are being annotated.
 *         uint annotations_off, // offset of the file to list of annotations for the method parameters. Format specified by annotation_set_ref_list.
 *        }
 *
 *        AnnotationSetRefList {
 *         uint size, // size of list in entries
 *         annotation_set_ref_item[size] list // element of the list ---|
 *        }                                                             |
 *                                                                      |
 *        --------------------------------------------------------------|
 *        |
 *        v
 *        uint annotations_off -----------------------------------------|
 *                                                                      |
 *                                                                      |
 *        --------------------------------------------------------------|
 *        |
 *        v
 *        AnnotationSetItem {
 *         uint size, // size of list in entries
 *         annotation_off_item[size] entries // element of the set -----|
 *        }                                                             |
 *                                                                      |
 *        --------------------------------------------------------------|
 *        |
 *        v
 *        uint annotation_off
 */

#ifndef DEX_ANNOTATIONS_HPP
#define DEX_ANNOTATIONS_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

#include "KUNAI/Utils/utils.hpp"
#include "KUNAI/Exceptions/exceptions.hpp"

namespace KUNAI
{
    namespace DEX
    {

        class ParameterAnnotation;
        class MethodAnnotations;
        class FieldAnnotation;
        class AnnotationsDirectoryItem;

        using parameterannotation_t = std::unique_ptr<ParameterAnnotation>;
        using parametersannotations_t = std::vector<parameterannotation_t>;

        using methodannotations_t = std::unique_ptr<MethodAnnotations>;
        using methodsannotations_t = std::vector<methodsannotations_t>;

        using fieldannotation_t = std::unique_ptr<FieldAnnotation>;
        using fieldsannotations_t = std::vector<fieldannotation_t>;

        using annotationsdirectoryitem_t = std::unique_ptr<AnnotationsDirectoryItem>;

        class ParameterAnnotation
        {
        public:
            ParameterAnnotation(std::uint32_t method_idx, std::uint32_t annotations_off);
            ~ParameterAnnotation() = default;

            std::uint32_t get_method_idx()
            {
                return method_idx;
            }

            std::uint32_t get_annotations_off()
            {
                return annotations_off;
            }

        private:
            std::uint32_t method_idx;
            std::uint32_t annotations_off;
        };

        class MethodAnnotations
        {
        public:
            MethodAnnotations(std::uint32_t method_idx, std::uint32_t annotations_off);
            ~MethodAnnotations() = default;

            std::uint32_t get_method_idx()
            {
                return method_idx;
            }

            std::uint32_t get_annotations_off()
            {
                return annotations_off;
            }

        private:
            std::uint32_t method_idx;
            std::uint32_t annotations_off;
        };

        class FieldAnnotation
        {
        public:
            FieldAnnotation(std::uint32_t field_idx, std::uint32_t annotations_off);
            ~FieldAnnotation() = default;

            std::uint32_t get_field_idx()
            {
                return field_idx;
            }

            std::uint32_t get_annotations_off()
            {
                return annotations_off;
            }

        private:
            std::uint32_t field_idx;
            std::uint32_t annotations_off;
        };

        class AnnotationsDirectoryItem
        {
        public:
            AnnotationsDirectoryItem(std::ifstream &input_file);
            ~AnnotationsDirectoryItem() = default;

            std::uint32_t get_class_annotations_off()
            {
                return class_annotations_off;
            }

            std::uint64_t get_fields_size()
            {
                return field_annotations.size();
            }

            FieldAnnotation *get_field_annotation_by_pos(std::uint64_t pos);

            const fieldsannotations_t &get_field_annotations() const
            {
                return field_annotations;
            }

            std::uint64_t get_annotated_methods_size()
            {
                return method_annotations.size();
            }

            MethodAnnotations *get_method_annotation_by_pos(std::uint64_t pos);

            const methodsannotations_t &get_method_annotations() const
            {
                return method_annotations;
            }

            std::uint64_t get_annotated_parameters_size()
            {
                return parameter_annotations.size();
            }

            ParameterAnnotation *get_parameter_annotation_by_pos(std::uint64_t pos);

            const parametersannotations_t &get_parameter_annotations() const
            {
                return parameter_annotations;
            }

        private:
            bool parse_annotations_directory_item(std::ifstream &input_file);

            std::uint32_t class_annotations_off;
            fieldsannotations_t field_annotations;
            methodsannotations_t method_annotations;
            parametersannotations_t parameter_annotations;
        };
    }
}

#endif