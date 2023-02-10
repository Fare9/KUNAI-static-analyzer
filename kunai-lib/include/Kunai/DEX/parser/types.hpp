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
        type_e get_type() const
        {
            return FUNDAMENTAL;
        }

        /// @brief Return a string with the name of the type
        /// @return string with the type
        std::string print_type() const
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
            DVMType(CLASS, name), name(name)
        {}

        /// @brief default destructor of DVMClass
        ~DVMClass() = default;

        /// @brief get type_e enum value for Class object.
        /// @return type_e with value CLASS.
        type_e get_type() const
        {
            return CLASS;
        }

        /// @brief get the type in string format
        /// @return type in string format
        std::string print_type() const
        {
            return "Class";
        }

        /// @brief Get the name of the class
        /// @return name of the class
        std::string& get_name()
        {
            return name;
        }
    };

    using dvmtype_t = std::unique_ptr<DVMType>;

    /// @brief Class that represent the array types
    class DVMArray : public DVMType
    {
        
        /// @brief depth of the array, it is possible to
        /// create arrays with different depth like [[C
        size_t depth;
        /// @brief type of the array
        dvmtype_t array_type;
    public:
        DVMArray(std::string raw, size_t depth, dvmtype_t& array) :
            DVMType(ARRAY, raw), depth(depth), array_type(std::move(array))
        {}

        
    };
} // namespace DEX
} // namespace KUNAI


#endif