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

#include <iostream>
#include <vector>
#include <unordered_map>

namespace KUNAI
{
namespace DEX
{
    /// @brief Class that contains the fields and methods from a class
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
    };
} // namespace DEX
} // namespace KUNAI


#endif