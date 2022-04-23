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

#include "utils.hpp"
#include "exceptions.hpp"

namespace KUNAI
{
    namespace DEX
    {

        class ParameterAnnotation;

        using parameterannotation_t = std::shared_ptr<ParameterAnnotation>;

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

        class MethodAnnotations;

        using methodannotations_t = std::shared_ptr<MethodAnnotations>;

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

        class FieldAnnotation;

        using fieldannotation_t = std::shared_ptr<FieldAnnotation>;

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

        class AnnotationsDirectoryItem;

        using annotationsdirectoryitem_t = std::shared_ptr<AnnotationsDirectoryItem>;

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

            fieldannotation_t get_field_annotation_by_pos(std::uint64_t pos);

            std::uint64_t get_annotated_methods_size()
            {
                return method_annotations.size();
            }

            methodannotations_t get_method_annotation_by_pos(std::uint64_t pos);

            std::uint64_t get_annotated_parameters_size()
            {
                return parameter_annotations.size();
            }

            parameterannotation_t get_parameter_annotation_by_pos(std::uint64_t pos);

        private:
            bool parse_annotations_directory_item(std::ifstream &input_file);

            std::uint32_t class_annotations_off;
            std::vector<fieldannotation_t> field_annotations;
            std::vector<methodannotations_t> method_annotations;
            std::vector<parameterannotation_t> parameter_annotations;
        };
    }
}

#endif