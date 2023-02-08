//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dvm_types.hpp
// @brief Useful types for working with the DVM
#ifndef KUNAI_DEX_DVM_DVM_TYPES_HPP
#define KUNAI_DEX_DVM_DVM_TYPES_HPP

#include <iostream>
#include <unordered_map>

namespace KUNAI
{
    namespace DEX
    {
        static const std::uint32_t ENDIAN_CONSTANT = 0x12345678;
        static const std::uint32_t REVERSE_ENDIAN_CONSTANT = 0x78563412;
        static const std::uint32_t NO_INDEX = 0xFFFFFFFF;

        static const std::uint8_t dex_magic[] = {'d', 'e', 'x', '\n'};
        static const std::uint8_t dex_magic_035[] = {'d', 'e', 'x', '\n', '0', '3', '5', '\0'};
        static const std::uint8_t dex_magic_037[] = {'d', 'e', 'x', '\n', '0', '3', '7', '\0'};
        static const std::uint8_t dex_magic_038[] = {'d', 'e', 'x', '\n', '0', '3', '8', '\0'};
        static const std::uint8_t dex_magic_039[] = {'d', 'e', 'x', '\n', '0', '3', '9', '\0'};

        namespace TYPES
        {
            /// @brief Identify the kind of argument inside
            /// a Dalvik Instruction
            enum class Kind
            {
                METH = 0,          //! method reference
                STRING = 1,        //! string index
                FIELD = 2,         //! field reference
                TYPE = 3,          //! type reference
                PROTO = 9,         //! prototype reference
                METH_PROTO = 10,   //! method reference and proto reference
                CALL_SITE = 11,    //! call site item
                VARIES = 4,        //!
                INLINE_METHOD = 5, //! inlined method
                VTABLE_OFFSET = 6, //! static linked
                FIELD_OFFSET = 7,  //! offset of a field (not reference)
                RAW_STRING = 8,    //!
                NONE_KIND = 99,    //!
            };

            /// @brief Identify different type of operations
            /// from instructions like branching, break, write
            /// or read.
            enum class Operation
            {
                CONDITIONAL_BRANCH_DVM_OPCODE = 0, //! conditional branch instructions ["throw", "throw.", "if."]
                UNCONDITIONAL_BRANCH_DVM_OPCODE,   //! unconditional branch instructions ["goto", "goto."]
                RET_BRANCH_DVM_OPCODE,             //! return instructions ["return", "return."]
                MULTI_BRANCH_DVM_OPCODE,           //! multi branching (switch) ["packed-switch$", "sparse-switch$"]
                CALL_DVM_OPCODE,                   //! call an external or internal method ["invoke", "invoke."]
                DATA_MOVEMENT_DVM_OPCODE,          //! move data instruction ["move", "move."]
                FIELD_READ_DVM_OPCODE,             //! read a field instruction [".get"]
                FIELD_WRITE_DVM_OPCODE,            //! write a field instruction [".put"]
                NONE_OPCODE = 99                   //!
            };

            /// @brief Enumeration used for operand type of opcodes
            enum class Operand
            {
                REGISTER = 0, //! register operand
                LITERAL = 1,  //! literal value
                RAW = 2,      //! raw value
                OFFSET = 3,   //! offset value

                KIND = 0x100, //! used together with others
            };

            /// @brief Access flags used in class_def_item,
            /// encoded_field, encoded_method and InnerClass
            /// https://source.android.com/devices/tech/dalvik/dex-format#access-flags
            enum class access_flags
            {
                NONE = 0x0,                         //! No access flags
                ACC_PUBLIC = 0x1,                   //! public type
                ACC_PRIVATE = 0x2,                  //! private type
                ACC_PROTECTED = 0x4,                //! protected type
                ACC_STATIC = 0x8,                   //! static (global) type
                ACC_FINAL = 0x10,                   //! final type (constant)
                ACC_SYNCHRONIZED = 0x20,            //! synchronized
                ACC_VOLATILE = 0x40,                //! Java volatile
                ACC_BRIDGE = 0x40,                  //!
                ACC_TRANSIENT = 0x80,               //!
                ACC_VARARGS = 0x80,                 //!
                ACC_NATIVE = 0x100,                 //! native type
                ACC_INTERFACE = 0x200,              //! interface type
                ACC_ABSTRACT = 0x400,               //! abstract type
                ACC_STRICT = 0x800,                 //!
                ACC_SYNTHETIC = 0x1000,             //!
                ACC_ANNOTATION = 0x2000,            //!
                ACC_ENUM = 0x4000,                  //! enum type
                UNUSED = 0x8000,                    //!
                ACC_CONSTRUCTOR = 0x10000,          //! constructor type
                ACC_DECLARED_SYNCHRONIZED = 0x20000 //!
            };

            /// @brief Enumeration used for the types.
            enum class value_format
            {
                VALUE_BYTE = 0x0,           //! ubyte[1]
                VALUE_SHORT = 0x2,          //! ubyte[size]
                VALUE_CHAR = 0x3,           //! ubyte[size]
                VALUE_INT = 0x4,            //! ubyte[size]
                VALUE_LONG = 0x6,           //! ubyte[size]
                VALUE_FLOAT = 0x10,         //! ubyte[size]
                VALUE_DOUBLE = 0x11,        //! ubyte[size]
                VALUE_METHOD_TYPE = 0x15,   //! ubyte[size]
                VALUE_METHOD_HANDLE = 0x16, //! ubyte[size]
                VALUE_STRING = 0x17,        //! ubyte[size]
                VALUE_TYPE = 0x18,          //! ubyte[size]
                VALUE_FIELD = 0x19,         //! ubyte[size]
                VALUE_METHOD = 0x1A,        //! ubyte[size]
                VALUE_ENUM = 0x1B,          //! ubyte[size]
                VALUE_ARRAY = 0x1C,         //! EncodedArray
                VALUE_ANNOTATION = 0x1D,    //! EncodedAnnotation
                VALUE_NULL = 0x1E,          //! None
                VALUE_BOOLEAN = 0x1F        //! None
            };

            /// @brief Opcodes from Dalvik Virtual Machine
            enum class opcodes
            {
#define OPCODE(ID, VAL) \
    ID = VAL,
#include "Kunai/DEX/DVM/dvm_types.def"
            };

            /// @brief String representation for all kind enum values
            static const std::unordered_map<Kind, std::string> KindString =
                {
                    {Kind::METH, "METH"},
                    {Kind::STRING, "STRING"},
                    {Kind::TYPE, "TYPE"},
                    {Kind::PROTO, "PROTO"},
                    {Kind::METH_PROTO, "METH_PROTO"},
                    {Kind::CALL_SITE, "CALL_SITE"},
                    {Kind::VARIES, "VARIES"},
                    {Kind::INLINE_METHOD, "INLINE_METHOD"},
                    {Kind::VTABLE_OFFSET, "VTABLE_OFFSET"},
                    {Kind::FIELD_OFFSET, "FIELD_OFFSET"},
                    {Kind::RAW_STRING, "RAW_STRING"},
                    {Kind::NONE_KIND, "NONE"}};

            /// @brief string representation for the access flags.
            static const std::unordered_map<access_flags, std::string> ACCESS_FLAGS_STR =
                {
                    {access_flags::ACC_PUBLIC, "public"},
                    {access_flags::ACC_PRIVATE, "private"},
                    {access_flags::ACC_PROTECTED, "protected"},
                    {access_flags::ACC_STATIC, "static"},
                    {access_flags::ACC_FINAL, "final"},
                    {access_flags::ACC_SYNCHRONIZED, "synchronized"},
                    {access_flags::ACC_BRIDGE, "bridge"},
                    {access_flags::ACC_VARARGS, "varargs"},
                    {access_flags::ACC_NATIVE, "native"},
                    {access_flags::ACC_INTERFACE, "interface"},
                    {access_flags::ACC_ABSTRACT, "abstract"},
                    {access_flags::ACC_STRICT, "strictfp"},
                    {access_flags::ACC_SYNTHETIC, "synthetic"},
                    {access_flags::ACC_ENUM, "enum"},
                    {access_flags::UNUSED, "unused"},
                    {access_flags::ACC_CONSTRUCTOR, "constructor"},
                    {access_flags::ACC_DECLARED_SYNCHRONIZED, "synchronized"}};

            /// @brief String name from a char for basic types
            static const std::unordered_map<char, std::string> type_descriptor_map =
                {
                    {'V', "void"},
                    {'Z', "boolean"},
                    {'B', "byte"},
                    {'S', "short"},
                    {'C', "char"},
                    {'I', "int"},
                    {'J', "long"},
                    {'F', "float"},
                    {'D', "double"}};
        } // namespace TYPES

    } // namespace DEX

} // namespace KUNAI

#endif