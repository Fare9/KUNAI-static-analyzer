//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file fields.hpp
// @brief Manage all the data from the fields from the DEX files.
#ifndef KUNAI_DEX_PARSER_METHODS_HPP
#define KUNAI_DEX_PARSER_METHODS_HPP

#include "Kunai/DEX/parser/protos.hpp"
#include "Kunai/DEX/parser/strings.hpp"
#include "Kunai/Utils/kunaistream.hpp"

#include <memory>
#include <vector>


namespace KUNAI
{
    namespace DEX
    {
        /// @brief forward declaration to keep a pointer on it
        class EncodedMethod;

        /// @brief MethodID represents a single method from DEX file
        class MethodID 
        {
            /// @brief Class where the method is
            DVMType* class_;
            /// @brief prototype of the method
            ProtoID* proto_;
            /// @brief name of the method
            std::string& name_;
            /// @brief pretty name with all the information
            std::string pretty_name;
            /// @brief pointer to the encoded method
            EncodedMethod * encoded_method;
        public:
            
            /// @brief Constructor of the MethodID
            /// @param class_ class of the method
            /// @param return_type type returned by prototype of the method
            /// @param name_ name of the method
            MethodID(DVMType* class_, ProtoID* proto_, std::string& name_)
                : class_(class_), proto_(proto_), name_(name_)
            {
            }

            /// @brief Destructor of the MethodID
            ~MethodID() = default;

            /// @brief get the encoded method set in this MethodID*
            /// @return the encoded method from this method
            EncodedMethod * get_encoded_method()
            {
                return encoded_method;
            }

            /// @brief Set the encoded method from this MethodID
            /// @param encoded_method proper encoded method
            void set_encoded_method(EncodedMethod * e)
            {
                encoded_method = e;
            }

            /// @brief Get a constant pointer to the class where method is
            /// @return constant pointer to class_
            const DVMType* get_class() const
            {
                return class_;
            }

            /// @brief Get a pointer to the class where method is
            /// @return pointer to class_
            DVMType* get_class()
            {
                return class_;
            }

            /// @brief Get a constant pointer to the proto of the method
            /// @return constant pointer to proto_
            const ProtoID* get_proto() const
            {
                return proto_;
            }

            /// @brief Get a pointer to the proto of the method
            /// @return pointer to proto_
            ProtoID* get_proto()
            {
                return proto_;
            }

            /// @brief Get a constant reference to the name of the method
            /// @return constant reference to name_
            const std::string& get_name() const
            {
                return name_;
            }

            /// @brief Get a reference to the name of the method
            /// @return reference to name_
            std::string& get_name()
            {
                return name_;
            }

            /// @brief Get a string representation of the method
            /// @return reference to pretty print of method
            std::string& pretty_method();
        };

        using methodid_t = std::unique_ptr<MethodID>;

        /// @brief Methods contains all the MethodIDs from the DEX file
        class Methods
        {
            /// @brief vector for containing the methods
            std::vector<methodid_t> methods;

            /// @brief number of methods
            std::uint32_t methods_size;
        public:

            /// @brief Constructor of Methods, default
            Methods() = default;

            /// @brief Destructor of Methods, default
            ~Methods() = default;

            /// @brief Parse all the method ids objects.
            /// @param stream stream with the dex file
            /// @param types types objects
            /// @param strings strings objects
            /// @param methods_offset offset to the ids of the methods
            /// @param methods_size number of methods to read
            void parse_methods(
                stream::KunaiStream* stream,
                Types* types,
                Protos* protos,
                Strings* strings,
                std::uint32_t methods_offset,
                std::uint32_t methods_size
            );

            /// @brief Get a constant reference to all the methods
            /// @return constant reference to vector of method ids
            const std::vector<methodid_t>& get_methods() const
            {
                return methods;
            }

            /// @brief Get a reference to all the methods
            /// @return reference to vector of method ids
            std::vector<methodid_t>& get_methods()
            {
                return methods;
            }

            /// @brief Get one of the methods by its position
            /// @param pos position in the vector
            /// @return pointer to MethodID
            MethodID* get_method(std::uint32_t pos);

            /// @brief Get the number of the methods
            /// @return value of methods_size
            std::uint32_t get_number_of_methods() const
            {
                return methods_size;
            }

            /// @brief Give a pretty print result of the methods
            /// @param os stream where to print methods
            /// @param entry entry to print
            /// @return stream
            friend std::ostream& operator<<(std::ostream& os, const Methods& entry);

            /// @brief Print the methods into an XML format.
            /// @param fos file where to dump it
            void to_xml(std::ofstream& fos);
        };

    } // namespace DEX
} // namespace KUNAI

#endif