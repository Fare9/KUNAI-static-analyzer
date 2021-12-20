#include "dex_annotations.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * ParameterAnnotation
         */
        ParameterAnnotation::ParameterAnnotation(std::uint32_t method_idx, std::uint32_t annotations_off) : method_idx(method_idx),
                                                                                                            annotations_off(annotations_off)
        {
        }

        /**
         * MethodAnnotations
         */
        MethodAnnotations::MethodAnnotations(std::uint32_t method_idx, std::uint32_t annotations_off) : method_idx(method_idx),
                                                                                                        annotations_off(annotations_off)
        {
        }

        /**
         * FieldAnnotation
         */
        FieldAnnotation::FieldAnnotation(std::uint32_t field_idx, std::uint32_t annotations_off) : field_idx(field_idx),
                                                                                                   annotations_off(annotations_off)
        {
        }

        /**
         * AnnotationsDirectoryItem
         */
        AnnotationsDirectoryItem::AnnotationsDirectoryItem(std::ifstream &input_file)
        {
            if (!parse_annotations_directory_item(input_file))
                throw exceptions::ParserReadingException("Error reading AnnotationsDirectoryItem");
        }

        std::shared_ptr<FieldAnnotation> AnnotationsDirectoryItem::get_field_annotation_by_pos(std::uint64_t pos)
        {
            if (pos >= field_annotations.size())
                return nullptr;
            return field_annotations[pos];
        }

        std::shared_ptr<MethodAnnotations> AnnotationsDirectoryItem::get_method_annotation_by_pos(std::uint64_t pos)
        {
            if (pos >= method_annotations.size())
                return nullptr;
            return method_annotations[pos];
        }

        std::shared_ptr<ParameterAnnotation> AnnotationsDirectoryItem::get_parameter_annotation_by_pos(std::uint64_t pos)
        {
            if (pos >= parameter_annotations.size())
                return nullptr;
            return parameter_annotations[pos];
        }

        bool AnnotationsDirectoryItem::parse_annotations_directory_item(std::ifstream &input_file)
        {
            auto current_offset = input_file.tellg();
            std::uint32_t class_annotations_off;
            std::uint32_t fields_size;
            std::uint32_t annotated_methods_size;
            std::uint32_t annotated_parameters_size;
            size_t i;

            std::uint32_t field_idx, method_idx, annotations_off;

            std::shared_ptr<FieldAnnotation> field_annotation;
            std::shared_ptr<MethodAnnotations> method_annotation;
            std::shared_ptr<ParameterAnnotation> parameter_annotation;

            if (!KUNAI::read_data_file<std::uint32_t>(class_annotations_off, sizeof(std::uint32_t), input_file))
                return false;

            this->class_annotations_off = class_annotations_off;

            if (!KUNAI::read_data_file<std::uint32_t>(fields_size, sizeof(std::uint32_t), input_file))
                return false;

            if (!KUNAI::read_data_file<std::uint32_t>(annotated_methods_size, sizeof(std::uint32_t), input_file))
                return false;

            if (!KUNAI::read_data_file<std::uint32_t>(annotated_parameters_size, sizeof(std::uint32_t), input_file))
                return false;

            for (i = 0; i < fields_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint32_t>(field_idx, sizeof(std::uint32_t), input_file))
                    return false;

                if (!KUNAI::read_data_file<std::uint32_t>(annotations_off, sizeof(std::uint32_t), input_file))
                    return false;

                field_annotation = std::make_shared<FieldAnnotation>(field_idx, annotations_off);
                field_annotations.push_back(field_annotation);
            }

            for (i = 0; i < annotated_methods_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint32_t>(method_idx, sizeof(std::uint32_t), input_file))
                    return false;

                if (!KUNAI::read_data_file<std::uint32_t>(annotations_off, sizeof(std::uint32_t), input_file))
                    return false;

                method_annotation = std::make_shared<MethodAnnotations>(method_idx, annotations_off);
                method_annotations.push_back(method_annotation);
            }

            for (i = 0; i < annotated_parameters_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint32_t>(method_idx, sizeof(std::uint32_t), input_file))
                    return false;

                if (!KUNAI::read_data_file<std::uint32_t>(annotations_off, sizeof(std::uint32_t), input_file))
                    return false;

                parameter_annotation = std::make_shared<ParameterAnnotation>(method_idx, annotations_off);
                parameter_annotations.push_back(parameter_annotation);
            }

            input_file.seekg(current_offset);
            return true;
        }
    }
}