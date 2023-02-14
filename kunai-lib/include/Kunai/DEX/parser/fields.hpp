//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file fields.hpp
// @brief Manage all the data from the fields from the DEX files.
#ifndef KUNAI_DEX_PARSER_FIELDS_HPP
#define KUNAI_DEX_PARSER_FIELDS_HPP

#include "Kunai/DEX/parser/types.hpp"
#include "Kunai/DEX/parser/strings.hpp"
#include "Kunai/Utils/kunaistream.hpp"

#include <memory>
#include <vector>

namespace KUNAI
{
namespace DEX
{
    /// @brief FieldID represent one of the fields from the
    /// DEX file.
    class FieldID
    {
        /// @brief Class where the field is
        DVMType* class_;
        /// @brief Type of the field
        DVMType* type_;
        /// @brief name of the field
        std::string& name_;
        /// @brief pretty name with all the information
        std::string pretty_name;
    public:

        /// @brief Constructor of the FieldID
        /// @param class_ class of the field
        /// @param type_ type of the field
        /// @param name_ name of the field
        FieldID(DVMType* class_, DVMType* type_, std::string& name_)
            : class_(class_), type_(type_), name_(name_)
        {
        }

        /// @brief Destructor of the FieldID
        ~FieldID() = default;

        /// @brief Get a constant pointer to the class where field is
        /// @return constant pointer to class_
        const DVMType* get_class() const
        {
            return class_;
        }

        /// @brief Get a pointer to the class where field is
        /// @return pointer to class_
        DVMType* get_class()
        {
            return class_;
        }

        /// @brief Get a constant pointer to the type of the field
        /// @return constant pointer to type_
        const DVMType* get_type() const
        {
            return type_;
        }

        /// @brief Get a pointer to the type of the field
        /// @return pointer to type_
        DVMType* get_type()
        {
            return type_;
        }

        /// @brief Get a constant reference to the name of the field
        /// @return constant reference to name_
        const std::string& get_name() const
        {
            return name_;
        }

        /// @brief Get a reference to the name of the field
        /// @return reference to name_
        std::string& get_name()
        {
            return name_;
        }

        /// @brief Get a string representation of the field
        /// @return reference to pretty print of field
        std::string& pretty_field();
    };

    using fieldid_t = std::unique_ptr<FieldID>;

    /// @brief Fields will contain all the FieldID from the DEX file
    class Fields
    {
        /// @brief vector for containing the fields
        std::vector<fieldid_t> fields;

        /// @brief number of fields
        std::uint32_t fields_size;
    public:

        /// @brief Constructor of Fields, default
        Fields() = default;

        /// @brief Destructor of Fields, default
        ~Fields() = default;

        /// @brief Parse all the field ids objects.
        /// @param stream stream with the dex file
        /// @param types types objects
        /// @param strings strings objects
        /// @param fields_offset offset to the ids of the fields
        /// @param fields_size number of fields to read
        void parse_fields(
            stream::KunaiStream* stream,
            Types* types,
            Strings* strings,
            std::uint32_t fields_offset,
            std::uint32_t fields_size
        );

        /// @brief Get a constant reference to all the fields
        /// @return constant reference to vector of field ids
        const std::vector<fieldid_t>& get_fields() const
        {
            return fields;
        }

        /// @brief Get a reference to all the fields
        /// @return reference to vector of field ids
        std::vector<fieldid_t>& get_fields()
        {
            return fields;
        }

        /// @brief Get one of the fields by its position
        /// @param pos position in the vector
        /// @return pointer to FieldID
        FieldID* get_field(std::uint32_t pos);

        /// @brief Get the number of the fields
        /// @return value of fields_size
        std::uint32_t get_number_of_fields() const
        {
            return fields_size;
        }

        /// @brief Give a pretty print result of the fields
        /// @param os stream where to print fields
        /// @param entry entry to print
        /// @return stream
        friend std::ostream& operator<<(std::ostream& os, const Fields& entry);

        /// @brief Print the fields into an XML format.
        /// @param fos file where to dump it
        void to_xml(std::ofstream& fos);
    };
} // namespace DEX
} // namespace KUNAI


#endif