/***
 * File: dex_dvm_types.hpp
 * 
 * Dictionaries used in Dalvik,
 * taken from androguard project.
 * 
 */

#ifndef DEX_DVM_TYPES_HPP
#define DEX_DVM_TYPES_HPP

#include <iostream>
#include <map>

namespace KUNAI
{
    namespace DEX
    {
        static const std::uint8_t dex_magic_035[] = {'d', 'e', 'x', '\n', '0', '3', '5', '\0'};
        static const std::uint8_t dex_magic_037[] = {'d', 'e', 'x', '\n', '0', '3', '7', '\0'};
        static const std::uint8_t dex_magic_038[] = {'d', 'e', 'x', '\n', '0', '3', '8', '\0'};
        static const std::uint8_t dex_magic_039[] = {'d', 'e', 'x', '\n', '0', '3', '9', '\0'};

        class DVMTypes
        {
        public:
            static const std::uint32_t ENDIAN_CONSTANT = 0x12345678;
            static const std::uint32_t REVERSE_ENDIAN_CONSTANT = 0x78563412;
            static const std::uint32_t NO_INDEX = 0xFFFFFFFF;

            enum Kind
            /***
             * Identify kind of argument inside of
             * argument inside a Dalvik Instruction.
             */
            {
                METH = 0,        // method reference
                STRING = 1,      // string index
                FIELD = 2,       // field reference
                TYPE = 3,        // type reference
                PROTO = 9,       // prototype reference
                METH_PROTO = 10, // method reference and proto reference
                CALL_SITE = 11,  // call site item
                VARIES = 4,
                INLINE_METHOD = 5, // Inlined method
                VTABLE_OFFSET = 6, // static linked
                FIELD_OFFSET = 7,
                RAW_STRING = 8,
                NONE = 99
            };

            enum Operand
            /***
             * Enumeration used for operand type of opcodes
             */
            {
                REGISTER = 0,
                LITERAL = 1,
                RAW = 2,
                OFFSET = 3,

                KIND = 0x100 // used together with others
            };

            enum ACCESS_FLAGS
            /***
             * Access flags used in class_def_item, encoded_field, 
             * encoded_method and InnerClass
             * https://source.android.com/devices/tech/dalvik/dex-format#access-flags
             */
            {
                ACC_PUBLIC = 0x1,
                ACC_PRIVATE = 0x2,
                ACC_PROTECTED = 0x4,
                ACC_STATIC = 0x8,
                ACC_FINAL = 0x10,
                ACC_SYNCHRONIZED = 0x20,
                ACC_VOLATILE = 0x40,
                ACC_BRIDGE = 0x40,
                ACC_TRANSIENT = 0x80,
                ACC_VARARGS = 0x80,
                ACC_NATIVE = 0x100,
                ACC_INTERFACE = 0x200,
                ACC_ABSTRACT = 0x400,
                ACC_STRICT = 0x800,
                ACC_SYNTHETIC = 0x1000,
                ACC_ANNOTATION = 0x2000,
                ACC_ENUM = 0x4000,
                UNUSED = 0x8000,
                ACC_CONSTRUCTOR = 0x10000,
                ACC_DECLARED_SYNCHRONIZED = 0x20000
            };

            enum VALUE_FORMATS
            {
                VALUE_BYTE = 0x0,           // ubyte[1]
                VALUE_SHORT = 0x2,          // ubyte[size]
                VALUE_CHAR = 0x3,           // ubyte[size]
                VALUE_INT = 0x4,            // ubyte[size]
                VALUE_LONG = 0x6,           // ubyte[size]
                VALUE_FLOAT = 0x10,         // ubyte[size]
                VALUE_DOUBLE = 0x11,        // ubyte[size]
                VALUE_METHOD_TYPE = 0x15,   // ubyte[size]
                VALUE_METHOD_HANDLE = 0x16, // ubyte[size]
                VALUE_STRING = 0x17,        // ubyte[size]
                VALUE_TYPE = 0x18,          // ubyte[size]
                VALUE_FIELD = 0x19,         // ubyte[size]
                VALUE_METHOD = 0x1A,        // ubyte[size]
                VALUE_ENUM = 0x1B,          // ubyte[size]
                VALUE_ARRAY = 0x1C,         // EncodedArray
                VALUE_ANNOTATION = 0x1D,    // EncodedAnnotation
                VALUE_NULL = 0x1E,          // None
                VALUE_BOOLEAN = 0x1F        // None
            };

            enum METHOD_HANDLE_TYPE_CODES
            {
                METHOD_HANDLE_TYPE_STATIC_PUT = 0x0,
                METHOD_HANDLE_TYPE_STATIC_GET,
                METHOD_HANDLE_TYPE_INSTANCE_PUT,
                METHOD_HANDLE_TYPE_INSTANCE_GET,
                METHOD_HANDLE_TYPE_INVOKE_STATIC,
                METHOD_HANDLE_TYPE_INVOKE_INSTANCE,
                METHOD_HANDLE_TYPE_INVOKE_CONSTRUCTOR,
                METHOD_HANDLE_TYPE_INVOKE_DIRECT,
                METHOD_HANDLE_TYPE_INVOKE_INTERFACE
            };
        };

                static std::map<DVMTypes::Kind, std::string> KindString = {
            {DVMTypes::METH, "METH"},
            {DVMTypes::STRING, "STRING"},
            {DVMTypes::FIELD, "FIELD"},
            {DVMTypes::TYPE, "TYPE"},
            {DVMTypes::PROTO, "PROTO"},
            {DVMTypes::METH_PROTO, "METH_PROTO"},
            {DVMTypes::CALL_SITE, "CALL_SITE"},
            {DVMTypes::VARIES, "VARIES"},
            {DVMTypes::INLINE_METHOD, "INLINE_METHOD"},
            {DVMTypes::VTABLE_OFFSET, "VTABLE_OFFSET"},
            {DVMTypes::FIELD_OFFSET, "FIELD_OFFSET"},
            {DVMTypes::RAW_STRING, "RAW_STRING"},
            {DVMTypes::NONE, "NONE"}};

        static std::map<DVMTypes::ACCESS_FLAGS, std::string> ACCESS_FLAGS_STR = {
            {DVMTypes::ACC_PUBLIC, "public"},
            {DVMTypes::ACC_PRIVATE, "private"},
            {DVMTypes::ACC_PROTECTED, "protected"},
            {DVMTypes::ACC_STATIC, "static"},
            {DVMTypes::ACC_FINAL, "final"},
            {DVMTypes::ACC_SYNCHRONIZED, "synchronized"},
            {DVMTypes::ACC_BRIDGE, "bridge"},
            {DVMTypes::ACC_VARARGS, "varargs"},
            {DVMTypes::ACC_NATIVE, "native"},
            {DVMTypes::ACC_INTERFACE, "interface"},
            {DVMTypes::ACC_ABSTRACT, "abstract"},
            {DVMTypes::ACC_STRICT, "strictfp"},
            {DVMTypes::ACC_SYNTHETIC, "synthetic"},
            {DVMTypes::ACC_ENUM, "enum"},
            {DVMTypes::UNUSED, "unused"},
            {DVMTypes::ACC_CONSTRUCTOR, "constructor"},
            {DVMTypes::ACC_DECLARED_SYNCHRONIZED, "synchronized"}};

        // already managed by dex_types
        static std::map<char, std::string> TYPE_DESCRIPTOR = {
            {'V', "void"},
            {'Z', "boolean"},
            {'B', "byte"},
            {'S', "short"},
            {'C', "char"},
            {'I', "int"},
            {'J', "long"},
            {'F', "float"},
            {'D', "double"}};
    }
}

#endif