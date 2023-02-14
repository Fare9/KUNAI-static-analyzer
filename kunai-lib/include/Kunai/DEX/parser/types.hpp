//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file types.hpp
// @brief Class for managing all the types from the DEX file
// the types are simply storing the string with the type.
#ifndef KUNAI_DEX_PARSER_TYPES_HPP
#define KUNAI_DEX_PARSER_TYPES_HPP

#include "Kunai/Utils/kunaistream.hpp"
#include "Kunai/DEX/parser/strings.hpp"

#include <vector>
#include <unordered_map>
#include <memory>

namespace KUNAI
{
namespace DEX
{
    /// @brief Represents the base class of a Type in the DVM
    /// we have different types
    class DVMType
    {
    public:
        /// @brief Types of the DVM we have by default fundamental,
        /// classes and array types
        enum type_e
        {
            FUNDAMENTAL,    //! fundamental type (int, float...)
            CLASS,          //! user defined class
            ARRAY,          //! an array type
            UNKNOWN         //! maybe wrong?
        };
    
    private:
        /// @brief what type is it?
        enum type_e type;
        /// @brief string with the type in raw
        std::string raw_type;

    public:
        /// @brief Constructor of DVMType
        /// @param type the type to overload
        /// @param raw_type string with the type in raw
        DVMType(type_e type, std::string raw_type)
            : type(type), raw_type(raw_type)
        {}

        /// @brief Destructor of DVMType
        virtual ~DVMType() = default;

        /// @brief Virtual method to return the type
        /// @return type of the variable
        virtual type_e get_type() const = 0;

        /// @brief Get the type on its string representation
        /// @return type in string format
        virtual std::string print_type() const = 0;
        
        /// @brief get raw string from the type object
        /// @return string from the type
        std::string& get_raw()
        {
            return raw_type;
        }
    };

    using dvmtype_t = std::unique_ptr<DVMType>;

    /// @brief Fundamental types from the DVM, these are the
    /// common from many other languages
    class DVMFundamental : public DVMType
    {
    public:
        /// @brief enum with the fundamental types
        enum fundamental_e
        {
            BOOLEAN,
            BYTE,
            CHAR,
            DOUBLE,
            FLOAT,
            INT,
            LONG,
            SHORT,
            VOID
        };

    private:
        /// @brief type of the fundamental
        enum fundamental_e f_type;
        /// @brief fundamental in string format
        std::string name;

        const std::unordered_map<fundamental_e, std::string> fundamental_s =
        {
            {BOOLEAN, "boolean"},
            {BYTE, "byte"},
            {CHAR, "char"},
            {DOUBLE, "double"},
            {FLOAT, "float"},
            {INT, "int"},
            {LONG, "long"},
            {SHORT, "short"},
            {VOID, "void"}
        };
    public:
        /// @brief Constructor of the DVMFundamental
        /// @param f_type enum of the fundamental type
        /// @param name name of the fundamental
        DVMFundamental(fundamental_e f_type, std::string name) :
            DVMType(FUNDAMENTAL, name), f_type(f_type), name(name)
        {}

        /// @brief Destructor of the fundamental
        ~DVMFundamental() = default;

        /// @brief get the type of the object
        /// @return return FUNDAMENTAL type
        type_e get_type() const override
        {
            return FUNDAMENTAL;
        }

        /// @brief Return a string with the name of the type
        /// @return string with the type
        std::string print_type() const override
        {
            return "Fundamental";
        }

        /// @brief get the stored fundamental type
        /// @return fundamental type enum
        fundamental_e get_fundamental_type() const
        {
            return f_type;
        }

        /// @brief Return a reference to a string with the fundamental type
        /// @param type fundamental value
        /// @return fundamental in string format
        const std::string& print_fundamental_type(fundamental_e type) const
        {
            return fundamental_s.at(type);
        }

        /// @brief Return a reference to the fundamental name
        /// @return fundamental name
        const std::string& get_name() const
        {
            return name;
        }
    };

    /// @brief Classes of the DVM
    class DVMClass : public DVMType
    {
        /// @brief name of the class
        std::string name;
    public:
        /// @brief constructor of DVM class with the name of the class
        /// @param name name of the class
        DVMClass(std::string name) :
            DVMType(CLASS, name), name(name.substr(1, name.size()-2))
        {}

        /// @brief default destructor of DVMClass
        ~DVMClass() = default;

        /// @brief get type_e enum value for Class object.
        /// @return type_e with value CLASS.
        type_e get_type() const override
        {
            return CLASS;
        }

        /// @brief get the type in string format
        /// @return type in string format
        std::string print_type() const override
        {
            return "Class";
        }

        /// @brief Get the name of the class
        /// @return name of the class
        const std::string& get_name() const
        {
            return name;
        }
        
        /// @brief Return a reference to the name
        /// @return name of the class
        std::string& get_name()
        {
            return name;
        }
    };

    /// @brief Class that represent the array types
    class DVMArray : public DVMType
    {
        
        /// @brief depth of the array, it is possible to
        /// create arrays with different depth like [[C
        size_t depth;
        /// @brief type of the array
        dvmtype_t array_type;
    public:
        /// @brief Constructor of DVMArray
        /// @param raw array type in raw
        /// @param depth how many depth the array contains
        /// @param array type of the array as a std::unique_ptr
        DVMArray(std::string raw, size_t depth, dvmtype_t& array) :
            DVMType(ARRAY, raw), depth(depth), array_type(std::move(array))
        {}

        /// @brief Destructor of DVMArray
        ~DVMArray() = default;

        /// @brief Return the type in this case an Array type
        /// @return ARRAY value from type_e enum
        type_e get_type() const override
        {
            return ARRAY;
        }

        /// @brief Return the string representation of the type
        /// @return ARRAY as a string
        std::string print_type() const override
        {
            return "Array";
        }

        /// @brief Return a pointer to the type of the array
        /// @return type of the array
        const DVMType* get_array_type() const
        {
            return array_type.get();
        }

        /// @brief Get the depth of the array specified as [[
        /// @return depth of the array
        size_t get_depth() const
        {
            return depth;
        }
    };

    /// @brief In case something unknown is found, we categorize it
    class Unknown : public DVMType
    {
    public:
        /// @brief Constructor of unknown type
        /// @param type type to be stored in parent class
        /// @param raw raw
        Unknown(std::string raw) :
            DVMType(UNKNOWN, raw)
        {}

        /// @brief Destructor of unknown type
        ~Unknown() = default;

        /// @brief Get Unkown type
        /// @return UNKNOWN value
        type_e get_type() const override
        {
            return UNKNOWN;
        }

        /// @brief Get Unkown type as a string
        /// @return UNKNOWN value as string
        std::string print_type() const override
        {
            return "Unknown";
        }
    };

    class Types
    {
        /// @brief types in the order they are parsed
        std::vector<dvmtype_t> ordered_types;
        /// @brief types by the id of the type
        std::unordered_map<std::uint32_t, DVMType*> types_by_id;
        //! @brief number of types according to header
        std::uint32_t number_of_types;
        //! @brief The offset where the types are
        std::uint32_t offset;
    public:
        /// @brief Constructor of the Types object, nothing for initialization
        Types() = default;
        /// @brief Destructor of Types
        ~Types() = default;

        /// @brief Parse the types and store them in the class
        /// @param stream stream with the DEX file
        /// @param strings strings to retrieve the type
        /// @param number_of_types number of types to retrieve
        /// @param types_offset offset where to read the types
        void parse_types(
            stream::KunaiStream* stream,
            Strings* strings,
            std::uint32_t number_of_types,
            std::uint32_t types_offset
        );

        /// @brief Get a reference to the vector with all the types
        /// @return constant reference to vector
        const std::vector<dvmtype_t>& get_ordered_types() const
        {
            return ordered_types;
        }

        /// @brief Get the types with the map of string id - type
        /// @return constant reference to map
        const std::unordered_map<std::uint32_t, DVMType*>& get_types_by_id() const
        {
            return types_by_id;
        }

        /// @brief Get a type given its string ID
        /// @param type_id ID of the type
        /// @return pointer to the type
        DVMType* get_type_by_id(std::uint32_t type_id);

        /// @brief Get a type given position
        /// @param pos position of the Type
        /// @return pointer to the type
        DVMType* get_type_from_order(std::uint32_t pos);

        /// @brief Get the number of types stored
        /// @return number of types
        std::uint32_t get_number_of_types() const
        {
            return number_of_types;
        }

        /// @brief Get the offset where types are stored
        /// @return offset where types are
        std::uint32_t get_offset() const
        {
            return offset;
        }

        /// @brief Pretty printer for Types
        /// @param os stream where to print it
        /// @param entry entry to print
        /// @return 
        friend std::ostream &operator<<(std::ostream &os, const Types &entry);

        /// @brief Dump the types to an xml file
        /// @param xml_file file where to dump content
        void to_xml(std::ofstream& xml_file);
    private:
        /// @brief Parse the given name in order to find what
        /// DEX type is
        /// @param name type from DEX
        /// @return object with the type
        dvmtype_t parse_type(std::string& name);
    };
} // namespace DEX
} // namespace KUNAI


#endif