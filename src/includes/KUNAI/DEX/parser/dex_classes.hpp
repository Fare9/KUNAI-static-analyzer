/***
 * @file dex_classes.hpp
 * @author @Farenain
 * 
 * @brief Android classes of the java code, composed
 *        by different fields:
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
 * 
 * interfaces_off points to:
 *  list of Class type
 * 
 * class_data_off points to:
 * 
 * ClassDataItem {
 *  uleb128 static_fields_size, // number of static fields defined in item
 *  uleb128 instance_fields_size, // number of instance fields defined in item
 *  uleb128 direct_methods_size, // number of direct methods defined in item
 *  uleb128 virtual_methods_size, // number of virtual methods defined in item
 *  encoded_field[static_fields_size] static_fields, // defined static fields, fields must be sorted by field_idx
 *  encoded_field[instance_fields_size] instance fields, // defined instance fields, fields must be sorted by field_idx
 *  encoded_method[direct_methods_size] direct_methods, // defined (static, private or constructor) methods. Must be sorted by method_idx in increasing order.
 *  encoded_method[virtual_methods_size] virtual_methods // defined virtual (static, private or constructor) methods. Must be sorted by method_idx in increasing order.
 * }
 */

#ifndef DEX_CLASSES_HPP
#define DEX_CLASSES_HPP

#include <iostream>
#include <map>

#include "dex_parents.hpp"

#include "dex_dvm_types.hpp"
#include "dex_types.hpp"
#include "dex_fields.hpp"
#include "dex_methods.hpp"
#include "dex_encoded.hpp"
#include "dex_annotations.hpp"
#include "exceptions.hpp"

namespace KUNAI {
    namespace DEX {
        
        class ClassDataItem
        {
        public:

            /**
             * @brief Constructor of ClassDataItem here the tool will parse
             *        fields and methods.
             * @param input_file: file to parse with DEX information.
             * @param file_size: size of file used for checking.
             * @param dex_fields: used for getting fields information during parsing.
             * @param dex_methods: used for getting information from methods during parsing.
             * @param dex_types: used for getting types information during parsing.
             * @return void
             */
            ClassDataItem(std::ifstream& input_file,
                            std::uint64_t file_size,
                            std::shared_ptr<DexFields> dex_fields,
                            std::shared_ptr<DexMethods> dex_methods,
                            std::shared_ptr<DexTypes> dex_types);
            /**
             * @brief ClassDataItem destructor
             * @return void
             */
            ~ClassDataItem() = default;

            /**
             * @brief Get the number of static fields in the class.
             * @return std::uint64_t
             */
            std::uint64_t get_number_of_static_fields()
            {
                return static_fields.size();
            }

            /**
             * @brief Get a class static field, by the field id, this is not a sorted number.
             * @param id: id of the field to retrieve.
             * @return std::shared_ptr<EncodedField>
             */
            std::shared_ptr<EncodedField> get_static_field_by_id(std::uint64_t id);
            
            /**
             * @brief Get a class static field by its position from parsing.
             * @param pos: position of static field.
             * @return std::shared_ptr<EncodedField>
             */
            std::shared_ptr<EncodedField> get_static_field_by_pos(std::uint64_t pos);

            /**
             * @brief return the number of instance fields from the class.
             * @return std::uint64_t
             */
            std::uint64_t get_number_of_instance_fields()
            {
                return instance_fields.size();
            }

            /**
             * @brief return a instance field from a class by the field id.
             * @param id: id of the EncodedField.
             * @return std::shared_ptr<EncodedField>
             */
            std::shared_ptr<EncodedField> get_instance_field_by_id(std::uint64_t id);
            
            /**
             * @brief return a instance field from a class by its position while parsing.
             * @param pos: position to retrieve.
             * @return std::shared_ptr<EncodedField>
             */
            std::shared_ptr<EncodedField> get_instance_field_by_pos(std::uint64_t pos);

            /**
             * @brief Get all the fields both static and instance fields.
             * @return std::vector<std::shared_ptr<EncodedField>>
             */
            std::vector<std::shared_ptr<EncodedField>> get_fields();

            /**
             * @brief Get the number of direct methods from the class.
             * @return std::uint64_t
             */
            std::uint64_t get_number_of_direct_methods()
            {
                return direct_methods.size();
            }

            /**
             * @brief Get a direct method by the id of the method.
             * @param id: id of the method to retrieve.
             * @return std::shared_ptr<EncodedMethod>
             */
            std::shared_ptr<EncodedMethod> get_direct_method_by_id(std::uint64_t id);
            
            /**
             * @brief Get a direct method by its position while parsing it.
             * @param pos: position of the method to retrieve.
             * @return std::shared_ptr<EncodedMethod>
             */
            std::shared_ptr<EncodedMethod> get_direct_method_by_pos(std::uint64_t pos);

            /**
             * @brief Get the number of virtual methods from the class.
             * @return std::uint64_t
             */
            std::uint64_t get_number_of_virtual_methods()
            {
                return virtual_methods.size();
            }

            /**
             * @brief Get a method from the class by its method id.
             * @param id: id of the method to retrieve.
             * @return std::shared_ptr<EncodedMethod>
             */
            std::shared_ptr<EncodedMethod> get_virtual_method_by_id(std::uint64_t id);
            
            /**
             * @brief Get a method from the class by its position while parsing.
             * @param pos: position of method while parsing.
             * @return std::shared_ptr<EncodedMethod>
             */
            std::shared_ptr<EncodedMethod> get_virtual_method_by_pos(std::uint64_t pos);

            /**
             * @brief Get all the methods object
             * 
             * @return std::vector<std::shared_ptr<EncodedMethod>> 
             */
            std::vector<std::shared_ptr<EncodedMethod>> get_methods();
        private:
            std::map<std::uint64_t, std::shared_ptr<EncodedField>> static_fields;
            std::map<std::uint64_t, std::shared_ptr<EncodedField>> instance_fields;
            std::map<std::uint64_t, std::shared_ptr<EncodedMethod>> direct_methods;
            std::map<std::uint64_t, std::shared_ptr<EncodedMethod>> virtual_methods;
        };

        class ClassDef : public ParentClass
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

            /**
             * @brief ClassDef constructor, parse class def by the structure classdef_t.
             * @param class_def: structure used to parse the class definition.
             * @param dex_str: strings object used while parsing.
             * @param dex_types: types object used while parsing.
             * @param dex_fields: fields object used while parsing.
             * @param dex_methods: methods object used while parsing.
             * @param input_file: DEX file where to read data.
             * @param file_size: size of the file for checks.
             * @return void
             */
            ClassDef(classdef_t class_def,
                     std::shared_ptr<DexStrings> dex_str, 
                     std::shared_ptr<DexTypes> dex_types,
                     std::shared_ptr<DexFields> dex_fields,
                     std::shared_ptr<DexMethods> dex_methods,
                     std::ifstream& input_file,
                     std::uint64_t file_size);
            /**
             * @brief Destructor of ClassDef.
             * @return void
             */
            ~ClassDef() = default;

            /**
             * @brief Get the ClassDef Class* object with class data.
             * @return Class*
             */
            Class* get_class_idx()
            {
                return class_idx.begin()->second;
            }

            /**
             * @brief return the access flags from the class.
             * @return DVMTypes::ACCESS_FLAGS
             */
            DVMTypes::ACCESS_FLAGS get_access_flags()
            {
                return access_flag;
            }

            /**
             * @brief Get the Class* object from the super class of current class.
             * @return Class*
             */
            Class* get_superclass_idx()
            {
                return superclass_idx.begin()->second;
            }

            /**
             * @brief Get the name of the file where the class is.
             * @return std::string*
             */
            std::string* get_source_file_idx()
            {
                return source_file_idx.begin()->second;
            }

            /**
             * @brief Get the number of interfaces this class implements
             * @return std::uint64_t
             */
            std::uint64_t get_number_of_interfaces()
            {
                return interfaces.size();
            }

            /**
             * @brief Get the Class* object from an interface by the class id.
             * @param id: id of the class.
             * @return Class*
             */
            Class* get_interface_by_class_id(std::uint16_t id);
            
            /**
             * @brief Get the Class* object from an interface by its position from parsing.
             * @param pos: position of interface from parsing.
             * @return Class*
             */
            Class* get_interface_by_pos(std::uint64_t pos);

            /**
             * @brief Get the ClassDataItem object from the ClassDef.
             * @return std::shared_ptr<ClassDataItem>
             */
            std::shared_ptr<ClassDataItem> get_class_data()
            {
                return class_data_items;
            }

        private:
            /**
             * @brief Private method for pasing the ClassDef object,
             *        get the interfaces, the annotations, the class_data_items
             *        and the static values.
             * @param input_file: DEX file where to read data.
             * @param file_size: size of the file for checks.
             * @param dex_str: strings object used while parsing.
             * @param dex_types: types object used while parsing.
             * @param dex_fields: fields object used while parsing.
             * @param dex_methods: methods object used while parsing.
             * @return bool
             */
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
            std::shared_ptr<AnnotationsDirectoryItem> annotation_directory_item;
            /**
             * classes def
             * 
             * offset to different fields and
             * methods
             * 
             * ClassDataItem:
             *  static_fields_size: uleb128
             *  instance_fields_size: uleb128
             *  direct_methods_size: uleb128
             *  virtual_methods_size: uleb128
             *  static_fields: EncodedField
             *  instance_fields: EncodedField
             *  direct_methods: EncodedMethod
             *  virtual_methods: EncodedMethod
             * 
             * EncodedField:
             *  field_id: uleb128
             *  access_flags: uleb128
             * EncodedMethod:
             *  method_id: uleb128
             *  access_flags: uleb128
             *  code_off: uleb128
             */
            std::uint32_t classess_off;
            std::shared_ptr<ClassDataItem> class_data_items;

            /**
             * static values
             * 
             * offset to different static variables.
             * 
             *  EncodedArrayItem
             *      size: uleb128
             *      values: EncodedValue[size]
             */
            std::uint32_t static_values_off;
            std::shared_ptr<EncodedArrayItem> static_values;
        };

        class DexClasses
        {
        public:
            /**
             * @brief Constructor of DexClasses this object manages all the classes from the DEX.
             * @param input_file: DEX file being parsed.
             * @param file_size: size of DEX file for checking.
             * @param number_of_classes: number of classes to parse from DEX.
             * @param offset: offset where the classes are.
             * @param dex_str: dex strings used while parsing.
             * @param dex_types: dex types used while parsing.
             * @param dex_fields: dex fields used while parsing.
             * @param dex_methods: dex methods used while parsing.
             * @return void
             */
            DexClasses(std::ifstream& input_file,
                        std::uint64_t file_size,
                        std::uint32_t number_of_classes,
                        std::uint32_t offset,
                        std::shared_ptr<DexStrings> dex_str,
                        std::shared_ptr<DexTypes> dex_types,
                        std::shared_ptr<DexFields> dex_fields,
                        std::shared_ptr<DexMethods> dex_methods);
            
            /**
             * @brief DexClasses destructor.
             * @return void
             */
            ~DexClasses();

            /**
             * @brief Get the number of classes parsed.
             * @return std::uint32_t
             */
            std::uint32_t get_number_of_classes()
            {
                return number_of_classes;
            }

            /**
             * @brief Get the ClassDef object by its position from parsing.
             * @return std::shared_ptr<ClassDef>
             */
            std::shared_ptr<ClassDef> get_class_by_pos(std::uint64_t pos);

            friend std::ostream& operator<<(std::ostream& os, const DexClasses& entry);
        private:
            /**
             * @brief private method for pasing the ClassDef information using the classdef_t structure.
             * @param input_file: dex file for parsing.
             * @param file_size: size of file for checks.
             * @return bool
             */
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