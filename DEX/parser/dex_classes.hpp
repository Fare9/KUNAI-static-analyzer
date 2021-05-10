/***
 * File: dex_classes.hpp
 * Author: @Farenain
 * 
 * Android classes of the java code, composed
 * by different fields:
 * 
 * ClassDefsStruct{
 *  uint class_idx, // index into type_ids list, must be a class type not array or primitive type.
 *  uint access_flags, // access flags for class (public, final, etc)
 *  uint superclass_idx, // index into type_ids list for super class, or constant value NO_INDEX if has no superclass.
 *  uint interfaces_off, // offset from start of the file to list of interfaces, or 0 if there are none. Offset should be in data section.
 *  uint source_file_idx, // index into string_ids for name of file containing original source, or value NO_INDEX if no information.
 *  uint annotations_off, // offset from start of file to annotations structure for this class, or 0 if no annotations.
 *  uint class_data_off, // offset from start of the file to the associated class data for this item, or 0 if no class data.
 *  uint static_values_off // offset from the start of the file to the list of initial values for static fields, or 0 if there are none.
 * }
 */

#ifndef DEX_CLASSES_HPP
#define DEX_CLASSES_HPP

#include <iostream>
#include <map>

#include "dex_dvm_types.hpp"
#include "dex_types.hpp"
#include "dex_fields.hpp"
#include "dex_methods.hpp"
#include "exceptions.hpp"

namespace KUNAI {
    namespace DEX {

        class EncodedField
        {
        public:
            EncodedField(FieldID* field_idx, std::uint64_t access_flags);
            ~EncodedField();

            FieldID* get_field();
            DVMTypes::ACCESS_FLAGS get_access_flags();

        private:
            FieldID* field_idx;
            DVMTypes::ACCESS_FLAGS access_flags;
        };

        class EncodedMethod
        {
        public:
            EncodedMethod(MethodID* method_id, std::uint64_t access_flags, std::uint64_t code_off);
            ~EncodedMethod();

            MethodID* get_method();
            DVMTypes::ACCESS_FLAGS get_access_flags();
            std::uint64_t get_code_offset();

        private:
            MethodID* method_id;
            DVMTypes::ACCESS_FLAGS access_flags;
            std::uint64_t code_off;
        };

        class ClassDataItem
        {
        public:
            ClassDataItem(std::ifstream& input_file,
                            std::shared_ptr<DexFields> dex_fields,
                            std::shared_ptr<DexMethods> dex_methods);
            ~ClassDataItem();

            std::uint64_t get_number_of_static_fields();
            std::shared_ptr<EncodedField> get_static_field_by_id(std::uint64_t id);
            std::shared_ptr<EncodedField> get_static_field_by_pos(std::uint64_t pos);

            std::uint64_t get_number_of_instance_fields();
            std::shared_ptr<EncodedField> get_instance_field_by_id(std::uint64_t id);
            std::shared_ptr<EncodedField> get_instance_field_by_pos(std::uint64_t pos);

            std::uint64_t get_number_of_direct_methods();
            std::shared_ptr<EncodedMethod> get_direct_method_by_id(std::uint64_t id);
            std::shared_ptr<EncodedMethod> get_direct_method_by_pos(std::uint64_t pos);

            std::uint64_t get_number_of_virtual_methods();
            std::shared_ptr<EncodedMethod> get_virtual_method_by_id(std::uint64_t id);
            std::shared_ptr<EncodedMethod> get_virtual_method_by_pos(std::uint64_t pos);
        private:
            std::map<std::uint64_t, std::shared_ptr<EncodedField>> static_fields;
            std::map<std::uint64_t, std::shared_ptr<EncodedField>> instance_fields;
            std::map<std::uint64_t, std::shared_ptr<EncodedMethod>> direct_methods;
            std::map<std::uint64_t, std::shared_ptr<EncodedMethod>> virtual_methods;
        };

        class ClassDef
        {
        public:
#pragma pack(1)
            struct classdef_t
            {
                std::uint32_t class_idx;
                std::uint32_t access_flags;
                std::uint32_t superclass_idx;
                std::uint32_t interfaces_off;
                std::uint32_t source_file_idx;
                std::uint32_t annotations_off;
                std::uint32_t class_data_off;
                std::uint32_t static_values_off;
            };
#pragma pack()

            ClassDef(classdef_t class_def,
                     std::shared_ptr<DexStrings> dex_str, 
                     std::shared_ptr<DexTypes> dex_types,
                     std::shared_ptr<DexFields> dex_fields,
                     std::shared_ptr<DexMethods> dex_methods,
                     std::ifstream& input_file,
                     std::uint64_t file_size);
            ~ClassDef();

            Class* get_class_idx();
            DVMTypes::ACCESS_FLAGS get_access_flags();
            Class* get_superclass_idx();
            std::string* get_source_file_idx();

            std::uint64_t get_number_of_interfaces();
            Class* get_interface_by_class_id(std::uint16_t id);
            Class* get_interface_by_pos(std::uint64_t pos);

            std::shared_ptr<ClassDataItem> get_class_data();

        private:
            bool parse_class_defs(std::ifstream& input_file, 
                                    std::uint64_t file_size, 
                                    std::shared_ptr<DexStrings> dex_str, 
                                    std::shared_ptr<DexTypes> dex_types,
                                    std::shared_ptr<DexFields> dex_fields,
                                    std::shared_ptr<DexMethods> dex_methods);


            std::map<std::uint32_t, Class*> class_idx;
            DVMTypes::ACCESS_FLAGS access_flag;
            std::map<std::uint32_t, Class*> superclass_idx;
            std::map<std::uint32_t, std::string*> source_file_idx;
            /**
             * type_list:
             *      size - uint size of list in entries
             *      list - type_item[size] element of list
             * type_item:
             *      ushort
             */
            std::uint32_t interfaces_off;
            std::map<std::uint16_t, Class*> interfaces;

            std::uint32_t annotations_off;

            /**
             * classes def
             * 
             * offset to different fields and
             * methods
             */
            std::uint32_t classess_off;
            std::shared_ptr<ClassDataItem> class_data_items;

            /**
             * static values
             */
            std::uint32_t static_values_off;
        };

        class DexClasses
        {
        public:
            DexClasses(std::ifstream& input_file,
                        std::uint64_t file_size,
                        std::uint32_t number_of_classes,
                        std::uint32_t offset,
                        std::shared_ptr<DexStrings> dex_str,
                        std::shared_ptr<DexTypes> dex_types,
                        std::shared_ptr<DexFields> dex_fields,
                        std::shared_ptr<DexMethods> dex_methods);
            ~DexClasses();

            std::uint32_t get_number_of_classes();
            std::shared_ptr<ClassDef> get_class_by_pos(std::uint64_t pos);
        private:
            bool parse_classes(std::ifstream& input_file, std::uint64_t file_size);

            std::uint32_t number_of_classes;
            std::uint32_t offset;
            std::shared_ptr<DexStrings> dex_str;
            std::shared_ptr<DexTypes> dex_types;
            std::shared_ptr<DexFields> dex_fields;
            std::shared_ptr<DexMethods> dex_methods;
            std::vector<std::shared_ptr<ClassDef>> class_defs;
        };
    }
}

#endif