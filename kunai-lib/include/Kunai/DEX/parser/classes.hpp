//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file classes.hpp
// @brief This is an important file, since classes manage all the magic
// behind the DEX files. Classes will contain all the encoded data
// (methods, fields, etc), as well as all the information about them
// two Classes are important here: `ClassDefsStruct` and `ClassDataItem`.
// `ClassDefsStruct` contains ids and offsets of the other information.
// while `ClassDataItem` contains all the items.

#ifndef KUNAI_DEX_PARSER_CLASSES_HPP
#define KUNAI_DEX_PARSER_CLASSES_HPP

#include "Kunai/DEX/parser/encoded.hpp"
#include "Kunai/DEX/parser/annotations.hpp"

#include <iostream>
#include <vector>
#include <unordered_map>

namespace KUNAI
{
namespace DEX
{
    /// @brief Class that contains the fields and methods from a class
    /// in a DEX file
    class ClassDataItem
    {
        /// @brief Static fields from the class
        std::vector<encodedfield_t> static_fields;
        std::unordered_map<std::uint64_t, EncodedField*> static_fields_by_id;

        /// @brief Instance fields from the class
        std::vector<encodedfield_t> instance_fields;
        std::unordered_map<std::uint64_t, EncodedField*> instance_fields_by_id;

        /// @brief Direct methods from the class
        std::vector<encodedmethod_t> direct_methods;
        std::unordered_map<std::uint64_t, EncodedMethod*> direct_methods_by_id;

        /// @brief Virtual methods from the class
        std::vector<encodedmethod_t> virtual_methods;
        std::unordered_map<std::uint64_t, EncodedMethod*> virtual_methods_by_id;

        /// @brief Vector with all the fields Static+Instance
        std::vector<EncodedField*> fields;
        /// @brief Vector with all the methods Direct+Virtual
        std::vector<EncodedMethod*> methods;
        

    public:
        /// @brief Constructor of ClassDataItem
        ClassDataItem() = default;
        /// @brief Destructor of ClassDataItem
        ~ClassDataItem() = default;

        /// @brief Method to parse the ClassDataItem
        /// @param stream stream DEX file
        /// @param fields fields of the DEX file
        /// @param methods methods of the DEX file
        /// @param types types of the DEX file
        void parse_class_data_item(
            stream::KunaiStream* stream,
            Fields* fields,
            Methods* methods,
            Types* types
        );

        /// @brief Get the number of the static fields
        /// @return size of static fields
        std::size_t get_number_of_static_fields() const
        {
            return static_fields.size();
        }

        /// @brief Get number of instance fields
        /// @return size of instance fields
        std::size_t get_number_of_instance_fields() const
        {
            return instance_fields.size();
        }

        /// @brief Get number of direct methods
        /// @return size of direct methods
        std::size_t get_number_of_direct_methods() const
        {
            return direct_methods.size();
        }

        /// @brief Get number of virtual methods
        /// @return size of virtual methods
        std::size_t get_number_of_virtual_methods() const
        {
            return virtual_methods.size();
        }

        /// @brief Get a pointer to static field from the reading order (recommended)
        /// @param ord order of the static field
        /// @return pointer to static encodedfield
        EncodedField* get_static_field_by_order(std::uint32_t ord);

        /// @brief Get a pointer to static field by the id of the FieldID
        /// @param id id of the FieldID
        /// @return pointer to static encodedfield
        EncodedField* get_static_field_by_id(std::uint32_t id);

        /// @brief Get an instance field from the reading order (recommended)
        /// @param ord order of the instance field
        /// @return pointer to instance encodedfield
        EncodedField* get_instance_field_by_order(std::uint32_t ord);

        /// @brief Get an instance field by the id of the FieldID
        /// @param id id of the FieldID
        /// @return pointer to instance encodedfield
        EncodedField* get_instance_field_by_id(std::uint32_t id);

        /// @brief Get a direct method from the reading order (recommended)
        /// @param ord order of the direct method
        /// @return pointer to direct encodedmethod
        EncodedMethod* get_direct_method_by_order(std::uint32_t ord);

        /// @brief Get a direct method by the id of the MethodID
        /// @param id id of the MethodID
        /// @return pointer to direct encodedmethod
        EncodedMethod* get_direct_method_by_id(std::uint32_t id);

        /// @brief Get a virtual method from the reading order (recommended)
        /// @param ord order of the virtual method
        /// @return pointer to virtual encodedmethod
        EncodedMethod* get_virtual_method_by_order(std::uint32_t ord);

        /// @brief Get a virtual method by the id of the MethodID
        /// @param id id of the MethodID
        /// @return pointer to virtual encodedmethod
        EncodedMethod* get_virtual_method_by_id(std::uint32_t id); 

        /// @brief Get all the fields from the class
        /// @return reference to vector with all the fields
        std::vector<EncodedField*>& get_fields(); 

        /// @brief Get all the methods from the class
        /// @return reference to vector with all the methods
        std::vector<EncodedMethod*>& get_methods();
    };

    /// @brief Definition of class with all the ids and offsets
    /// for all the other data
    class ClassDef
    {
    public:
#pragma pack(1)
        /// @brief Definition of offsets and IDs
        struct classdefstruct_t
        {
            std::uint32_t class_idx;            //! idx for the current class
            std::uint32_t access_flags;         //! flags for this class
            std::uint32_t superclass_idx;       //! parent class id
            std::uint32_t interfaces_off;       //! interfaces implemented by class
            std::uint32_t source_file_idx;      //! idx to a string with source file
            std::uint32_t annotations_off;      //! debugging information and other data
            std::uint32_t class_data_off;       //! offset to class data item
            std::uint32_t static_values_off;    //! offset to static values
        };
#pragma pack()

    private:
        /// @brief Structure with the definition of the class
        classdefstruct_t classdefstruct;
        /// @brief DVMClass for the current class
        DVMClass* class_idx;
        /// @brief DVMClass for the parent/super class
        DVMClass* superclass_idx;
        /// @brief String with the source file
        std::string source_file;

        /// @brief vector with the interfaces implemented
        std::vector<DVMClass*> interfaces;

        /// @brief Annotations of the class
        AnnotationDirectoryItem annotation_directory;

        /// @brief ClassDataItem value of the current class
        ClassDataItem class_data_item;

        /// @brief Array of initial values for static fields.
        std::vector<encodedarray_t> static_values;
    public:
        /// @brief Constructor of ClassDef
        ClassDef() = default;
        /// @brief Destructor of ClassDef
        ~ClassDef() = default;

        /// @brief Parse the current ClassDef for that we will parse the
        /// classdef_t structure, and then all the other fields.
        /// @param stream stream with DEX file currently parsed
        /// @param strings strings of the DEX file
        /// @param types types of the DEX file
        /// @param fields fields of the DEX file
        /// @param methods methods of the DEX file
        void parse_class_def(stream::KunaiStream* stream,
                             Strings* strings,
                             Types* types,
                             Fields* fields,
                             Methods* methods);

        /// @brief Get a constant reference to the classdefstruct_t
        /// of the class, this structure contains information about
        /// the class
        /// @return constant reference to classdefstruct_t structure
        const classdefstruct_t& get_class_def_struct() const
        {
            return classdefstruct;
        }

        /// @brief Get a reference to the classdefstruct_t
        /// of the class, this structure contains information about
        /// the class
        /// @return reference to classdefstruct_t structure
        classdefstruct_t& get_class_def_struct()
        {
            return classdefstruct;
        }

        /// @brief Get a pointer to the DVMClass of the current class
        /// @return pointer to DVMClass of current class
        DVMClass * get_class_idx()
        {
            return class_idx;
        }

        /// @brief Get the access flags of the current class
        /// @return 
        TYPES::access_flags get_access_flags() const
        {
            return static_cast<TYPES::access_flags>(classdefstruct.access_flags);
        }

        /// @brief Get a pointer to the DVMClass of the super class of the current one
        /// @return pointer to DVMClass of the super class
        DVMClass * get_superclass()
        {
            return superclass_idx;
        }

        /// @brief Get a constant reference to the string with the source file
        /// @return constant reference to source file string
        const std::string& get_source_file() const
        {
            return source_file;
        }

        /// @brief Get a constant reference to the vector with the interfaces implemented
        /// @return constant reference to interfaces
        const std::vector<DVMClass*>& get_interfaces() const
        {
            return interfaces;
        }

        /// @brief Get a reference to the vector with the interfaces implemented
        /// @return reference to interfaces
        std::vector<DVMClass*>& get_interfaces()
        {
            return interfaces;
        }

        /// @brief Get a constant reference to the class data item
        /// @return constant reference to the class data item
        const ClassDataItem& get_class_data_item() const
        {
            return class_data_item;
        }

        /// @brief Get a reference to the class data item
        /// @return reference to the class data item
        ClassDataItem& get_class_data_item()
        {
            return class_data_item;
        }
    

    };

    using classdef_t = std::unique_ptr<ClassDef>;

    /// @brief All classes from the DEX files
    class Classes
    {
        /// @brief All the class_defs from the DEX, one
        /// for each class
        std::vector<classdef_t> class_defs;
        /// @brief Number of classes
        std::uint32_t number_of_classes;
    public:
        /// @brief Constructor from Classes
        Classes() = default;
        /// @brief Destructor from Classes
        ~Classes() = default;
        /// @brief Parse all the classes from the DEX files
        /// @param stream stream with the DEX file
        /// @param number_of_classes number of classes from the DEX
        /// @param offset offset to parse the classes
        /// @param strings strings from the DEX file
        /// @param types types from the DEX file
        /// @param fields fields from the DEX file
        /// @param methods methods from the DEX file
        void parse_classes(
            stream::KunaiStream* stream,
            std::uint32_t number_of_classes,
            std::uint32_t offset,
            Strings* strings,
            Types* types,
            Fields* fields,
            Methods* methods 
        );

        /// @brief Get the number of the classes from the DEX file
        /// @return number of classes of DEX file
        std::uint32_t get_number_of_classes() const
        {
            return number_of_classes;
        }

        /// @brief Get a constant reference to all the class defs from the DEX
        /// @return constant reference to vector with class_defs
        const std::vector<classdef_t>& get_classdefs() const
        {
            return class_defs;
        }

        /// @brief Get a reference to all the class defs from the DEX
        /// @return reference to vector with class_defs
        std::vector<classdef_t>& get_classdefs()
        {
            return class_defs;
        }

        friend std::ostream& operator<<(std::ostream& os, const Classes& entry);
    };
} // namespace DEX
} // namespace KUNAI


#endif