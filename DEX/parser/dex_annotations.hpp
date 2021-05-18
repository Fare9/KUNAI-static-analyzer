/***
 * File: dex_annotations.hpp
 * Author: @Farenain
 * 
 * Android classes used to represent
 * annotations. These are pointed by
 * the offset annotations_off from ClassDefStruct.
 * 
 * annotations_off points to:
 * 
 * AnnotationsDirectoryItem {
 *  uint class_annotations_off, // offset to annotations made directly on the class.
 *  uint fields_size, // number of fields annotation by this item.
 *  uint annotated_methods_size, // number of methods annotated by this item.
 *  uint annotated_parameters_size, // number of method parameters list annotated by this item.
 *  field_annotation[fields_size] field_annotations, // associated field annotations. Must be field_idx sorted in increasing order.
 *  method_annotation[methods_size] method_annotations, // associated method annotations. Must be method_idx sorted in increasing order
 *  parameter_annotation[parameters_size] parameter_annotations //  associated method parameter annotations. Must be method_idx sorted in increasing order.
 * }
 * 
 * FieldAnnotation {
 *  uint field_idx, // index into field_ids
 *  uint annotations_off // offset of the file to the list of annotations for the field. Format specified by annotation_set_item.
 * }
 * 
 * MethodAnnotation {
 *  uint method_idx, // index into method_ids of the method being annotated.
 *  uint annotations_off, // offset of the file to list of annotations for the method. Format specified by annotation_set_item.
 * }
 * 
 * ParameterAnnotation {
 *  uint method_idx, // index into method_idx of the method whose parameters are being annotated.
 *  uint annotations_off, // offset of the file to list of annotations for the method parameters. Format specified by annotation_set_ref_list.
 * }
 * 
 * AnnotationSetRefList {
 *  uint size, // size of list in entries
 *  annotation_set_ref_item[size] list // element of the list ---|
 * }                                                             |
 *                                                               |
 * --------------------------------------------------------------|
 * |
 * v
 * uint annotations_off -----------------------------------------|
 *                                                               |
 *                                                               |
 * --------------------------------------------------------------|
 * |
 * v
 * AnnotationSetItem {
 *  uint size, // size of list in entries
 *  annotation_off_item[size] entries // element of the set -----|
 * }                                                             |
 *                                                               |
 * --------------------------------------------------------------|
 * |
 * v
 * uint annotation_off
 */


#ifndef DEX_ANNOTATIONS_HPP
#define DEX_ANNOTATIONS_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

#include "utils.hpp"
#include "exceptions.hpp"

namespace KUNAI {
    namespace DEX {

        class ParameterAnnotation
        {
        public:
            ParameterAnnotation(std::uint32_t method_idx, std::uint32_t annotations_off);
            ~ParameterAnnotation();

            std::uint32_t get_method_idx();
            std::uint32_t get_annotations_off();

        private:
            std::uint32_t method_idx;
            std::uint32_t annotations_off;
        };

        class MethodAnnotations
        {
        public:
            MethodAnnotations(std::uint32_t method_idx, std::uint32_t annotations_off);
            ~MethodAnnotations();

            std::uint32_t get_method_idx();
            std::uint32_t get_annotations_off();

        private:
            std::uint32_t method_idx;
            std::uint32_t annotations_off;
        };

        class FieldAnnotation
        {
        public:
            FieldAnnotation(std::uint32_t field_idx, std::uint32_t annotations_off);
            ~FieldAnnotation();

            std::uint32_t get_field_idx();
            std::uint32_t get_annotations_off();

        private:
            std::uint32_t field_idx;
            std::uint32_t annotations_off;
        };

        class AnnotationsDirectoryItem
        {
        public:
            AnnotationsDirectoryItem(std::ifstream& input_file);
            ~AnnotationsDirectoryItem();

            std::uint32_t get_class_annotations_off();
            std::uint64_t get_fields_size();
            std::shared_ptr<FieldAnnotation> get_field_annotation_by_pos(std::uint64_t pos);
            std::uint64_t get_annotated_methods_size();
            std::shared_ptr<MethodAnnotations> get_method_annotation_by_pos(std::uint64_t pos);
            std::uint64_t get_annotated_parameters_size();
            std::shared_ptr<ParameterAnnotation> get_parameter_annotation_by_pos(std::uint64_t pos);

        private:
            bool parse_annotations_directory_item(std::ifstream& input_file);
            
            std::uint32_t class_annotations_off;
            std::vector<std::shared_ptr<FieldAnnotation>> field_annotations;
            std::vector<std::shared_ptr<MethodAnnotations>> method_annotations;
            std::vector<std::shared_ptr<ParameterAnnotation>> parameter_annotations;
        };
    }
}

#endif