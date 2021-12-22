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
            /**
             * @brief Identify kind of argument inside of argument inside a Dalvik Instruction.
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
                NONE_KIND = 99
            };

            enum Operation
            /**
             * @brief Identify different type of operations from instructions like branching, break, write or read.
             */
            {
                BRANCH_DVM_OPCODE = 0,      // branch instructions ["throw", "throw.", "if.", "goto", "goto.", "return", "return.", "packed-switch$", "sparse-switch$"]
                BREAK_DVM_OPCODE = 1,       // break instructions ["invoke.", "move."]
                FIELD_READ_DVM_OPCODE = 2,  // read a field instruction [".get"]
                FIELD_WRITE_DVM_OPCODE = 3, // write a field instruction [".put"]
                NONE_OPCODE = 99
            };

            enum Operand
            /**
             * @brief Enumeration used for operand type of opcodes
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
             * @brief Access flags used in class_def_item, encoded_field, encoded_method and InnerClass https://source.android.com/devices/tech/dalvik/dex-format#access-flags
             */
            {
                NONE = 0x0,
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
            /**
             * @brief Enumeration used for the types.
             */
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
            /**
             * @brief types for method handle.
             */
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

            enum REF_TYPE
            /**
             * @brief Reference used in the cross ref of the classes, to store the type of reference to the class.
             */
            {
                REF_NEW_INSTANCE = 0x22,
                REF_CLASS_USAGE = 0x1c,
                INVOKE_VIRTUAL = 0x6e,
                INVOKE_SUPER = 0x6f,
                INVOKE_DIRECT = 0x70,
                INVOKE_STATIC = 0x71,
                INVOKE_INTERFACE = 0x72,
                INVOKE_VIRTUAL_RANGE = 0x74,
                INVOKE_SUPER_RANGE = 0x75,
                INVOKE_DIRECT_RANGE = 0x76,
                INVOKE_STATIC_RANGE = 0x77,
                INVOKE_INTERFACE_RANGE = 0x78
            };

            enum Opcode
            /**
             * @brief values for all the different opcodes.
             */
            {
                // BEGIN(libdex-opcode-enum); GENERATED AUTOMATICALLY BY opcode-gen
                OP_NOP = 0x00,
                OP_MOVE = 0x01,
                OP_MOVE_FROM16 = 0x02,
                OP_MOVE_16 = 0x03,
                OP_MOVE_WIDE = 0x04,
                OP_MOVE_WIDE_FROM16 = 0x05,
                OP_MOVE_WIDE_16 = 0x06,
                OP_MOVE_OBJECT = 0x07,
                OP_MOVE_OBJECT_FROM16 = 0x08,
                OP_MOVE_OBJECT_16 = 0x09,
                OP_MOVE_RESULT = 0x0a,
                OP_MOVE_RESULT_WIDE = 0x0b,
                OP_MOVE_RESULT_OBJECT = 0x0c,
                OP_MOVE_EXCEPTION = 0x0d,
                OP_RETURN_VOID = 0x0e,
                OP_RETURN = 0x0f,
                OP_RETURN_WIDE = 0x10,
                OP_RETURN_OBJECT = 0x11,
                OP_CONST_4 = 0x12,
                OP_CONST_16 = 0x13,
                OP_CONST = 0x14,
                OP_CONST_HIGH16 = 0x15,
                OP_CONST_WIDE_16 = 0x16,
                OP_CONST_WIDE_32 = 0x17,
                OP_CONST_WIDE = 0x18,
                OP_CONST_WIDE_HIGH16 = 0x19,
                OP_CONST_STRING = 0x1a,
                OP_CONST_STRING_JUMBO = 0x1b,
                OP_CONST_CLASS = 0x1c,
                OP_MONITOR_ENTER = 0x1d,
                OP_MONITOR_EXIT = 0x1e,
                OP_CHECK_CAST = 0x1f,
                OP_INSTANCE_OF = 0x20,
                OP_ARRAY_LENGTH = 0x21,
                OP_NEW_INSTANCE = 0x22,
                OP_NEW_ARRAY = 0x23,
                OP_FILLED_NEW_ARRAY = 0x24,
                OP_FILLED_NEW_ARRAY_RANGE = 0x25,
                OP_FILL_ARRAY_DATA = 0x26,
                OP_THROW = 0x27,
                OP_GOTO = 0x28,
                OP_GOTO_16 = 0x29,
                OP_GOTO_32 = 0x2a,
                OP_PACKED_SWITCH = 0x2b,
                OP_SPARSE_SWITCH = 0x2c,
                OP_CMPL_FLOAT = 0x2d,
                OP_CMPG_FLOAT = 0x2e,
                OP_CMPL_DOUBLE = 0x2f,
                OP_CMPG_DOUBLE = 0x30,
                OP_CMP_LONG = 0x31,
                OP_IF_EQ = 0x32,
                OP_IF_NE = 0x33,
                OP_IF_LT = 0x34,
                OP_IF_GE = 0x35,
                OP_IF_GT = 0x36,
                OP_IF_LE = 0x37,
                OP_IF_EQZ = 0x38,
                OP_IF_NEZ = 0x39,
                OP_IF_LTZ = 0x3a,
                OP_IF_GEZ = 0x3b,
                OP_IF_GTZ = 0x3c,
                OP_IF_LEZ = 0x3d,
                OP_UNUSED_3E = 0x3e,
                OP_UNUSED_3F = 0x3f,
                OP_UNUSED_40 = 0x40,
                OP_UNUSED_41 = 0x41,
                OP_UNUSED_42 = 0x42,
                OP_UNUSED_43 = 0x43,
                OP_AGET = 0x44,
                OP_AGET_WIDE = 0x45,
                OP_AGET_OBJECT = 0x46,
                OP_AGET_BOOLEAN = 0x47,
                OP_AGET_BYTE = 0x48,
                OP_AGET_CHAR = 0x49,
                OP_AGET_SHORT = 0x4a,
                OP_APUT = 0x4b,
                OP_APUT_WIDE = 0x4c,
                OP_APUT_OBJECT = 0x4d,
                OP_APUT_BOOLEAN = 0x4e,
                OP_APUT_BYTE = 0x4f,
                OP_APUT_CHAR = 0x50,
                OP_APUT_SHORT = 0x51,
                OP_IGET = 0x52,
                OP_IGET_WIDE = 0x53,
                OP_IGET_OBJECT = 0x54,
                OP_IGET_BOOLEAN = 0x55,
                OP_IGET_BYTE = 0x56,
                OP_IGET_CHAR = 0x57,
                OP_IGET_SHORT = 0x58,
                OP_IPUT = 0x59,
                OP_IPUT_WIDE = 0x5a,
                OP_IPUT_OBJECT = 0x5b,
                OP_IPUT_BOOLEAN = 0x5c,
                OP_IPUT_BYTE = 0x5d,
                OP_IPUT_CHAR = 0x5e,
                OP_IPUT_SHORT = 0x5f,
                OP_SGET = 0x60,
                OP_SGET_WIDE = 0x61,
                OP_SGET_OBJECT = 0x62,
                OP_SGET_BOOLEAN = 0x63,
                OP_SGET_BYTE = 0x64,
                OP_SGET_CHAR = 0x65,
                OP_SGET_SHORT = 0x66,
                OP_SPUT = 0x67,
                OP_SPUT_WIDE = 0x68,
                OP_SPUT_OBJECT = 0x69,
                OP_SPUT_BOOLEAN = 0x6a,
                OP_SPUT_BYTE = 0x6b,
                OP_SPUT_CHAR = 0x6c,
                OP_SPUT_SHORT = 0x6d,
                OP_INVOKE_VIRTUAL = 0x6e,
                OP_INVOKE_SUPER = 0x6f,
                OP_INVOKE_DIRECT = 0x70,
                OP_INVOKE_STATIC = 0x71,
                OP_INVOKE_INTERFACE = 0x72,
                OP_UNUSED_73 = 0x73,
                OP_INVOKE_VIRTUAL_RANGE = 0x74,
                OP_INVOKE_SUPER_RANGE = 0x75,
                OP_INVOKE_DIRECT_RANGE = 0x76,
                OP_INVOKE_STATIC_RANGE = 0x77,
                OP_INVOKE_INTERFACE_RANGE = 0x78,
                OP_UNUSED_79 = 0x79,
                OP_UNUSED_7A = 0x7a,
                OP_NEG_INT = 0x7b,
                OP_NOT_INT = 0x7c,
                OP_NEG_LONG = 0x7d,
                OP_NOT_LONG = 0x7e,
                OP_NEG_FLOAT = 0x7f,
                OP_NEG_DOUBLE = 0x80,
                OP_INT_TO_LONG = 0x81,
                OP_INT_TO_FLOAT = 0x82,
                OP_INT_TO_DOUBLE = 0x83,
                OP_LONG_TO_INT = 0x84,
                OP_LONG_TO_FLOAT = 0x85,
                OP_LONG_TO_DOUBLE = 0x86,
                OP_FLOAT_TO_INT = 0x87,
                OP_FLOAT_TO_LONG = 0x88,
                OP_FLOAT_TO_DOUBLE = 0x89,
                OP_DOUBLE_TO_INT = 0x8a,
                OP_DOUBLE_TO_LONG = 0x8b,
                OP_DOUBLE_TO_FLOAT = 0x8c,
                OP_INT_TO_BYTE = 0x8d,
                OP_INT_TO_CHAR = 0x8e,
                OP_INT_TO_SHORT = 0x8f,
                OP_ADD_INT = 0x90,
                OP_SUB_INT = 0x91,
                OP_MUL_INT = 0x92,
                OP_DIV_INT = 0x93,
                OP_REM_INT = 0x94,
                OP_AND_INT = 0x95,
                OP_OR_INT = 0x96,
                OP_XOR_INT = 0x97,
                OP_SHL_INT = 0x98,
                OP_SHR_INT = 0x99,
                OP_USHR_INT = 0x9a,
                OP_ADD_LONG = 0x9b,
                OP_SUB_LONG = 0x9c,
                OP_MUL_LONG = 0x9d,
                OP_DIV_LONG = 0x9e,
                OP_REM_LONG = 0x9f,
                OP_AND_LONG = 0xa0,
                OP_OR_LONG = 0xa1,
                OP_XOR_LONG = 0xa2,
                OP_SHL_LONG = 0xa3,
                OP_SHR_LONG = 0xa4,
                OP_USHR_LONG = 0xa5,
                OP_ADD_FLOAT = 0xa6,
                OP_SUB_FLOAT = 0xa7,
                OP_MUL_FLOAT = 0xa8,
                OP_DIV_FLOAT = 0xa9,
                OP_REM_FLOAT = 0xaa,
                OP_ADD_DOUBLE = 0xab,
                OP_SUB_DOUBLE = 0xac,
                OP_MUL_DOUBLE = 0xad,
                OP_DIV_DOUBLE = 0xae,
                OP_REM_DOUBLE = 0xaf,
                OP_ADD_INT_2ADDR = 0xb0,
                OP_SUB_INT_2ADDR = 0xb1,
                OP_MUL_INT_2ADDR = 0xb2,
                OP_DIV_INT_2ADDR = 0xb3,
                OP_REM_INT_2ADDR = 0xb4,
                OP_AND_INT_2ADDR = 0xb5,
                OP_OR_INT_2ADDR = 0xb6,
                OP_XOR_INT_2ADDR = 0xb7,
                OP_SHL_INT_2ADDR = 0xb8,
                OP_SHR_INT_2ADDR = 0xb9,
                OP_USHR_INT_2ADDR = 0xba,
                OP_ADD_LONG_2ADDR = 0xbb,
                OP_SUB_LONG_2ADDR = 0xbc,
                OP_MUL_LONG_2ADDR = 0xbd,
                OP_DIV_LONG_2ADDR = 0xbe,
                OP_REM_LONG_2ADDR = 0xbf,
                OP_AND_LONG_2ADDR = 0xc0,
                OP_OR_LONG_2ADDR = 0xc1,
                OP_XOR_LONG_2ADDR = 0xc2,
                OP_SHL_LONG_2ADDR = 0xc3,
                OP_SHR_LONG_2ADDR = 0xc4,
                OP_USHR_LONG_2ADDR = 0xc5,
                OP_ADD_FLOAT_2ADDR = 0xc6,
                OP_SUB_FLOAT_2ADDR = 0xc7,
                OP_MUL_FLOAT_2ADDR = 0xc8,
                OP_DIV_FLOAT_2ADDR = 0xc9,
                OP_REM_FLOAT_2ADDR = 0xca,
                OP_ADD_DOUBLE_2ADDR = 0xcb,
                OP_SUB_DOUBLE_2ADDR = 0xcc,
                OP_MUL_DOUBLE_2ADDR = 0xcd,
                OP_DIV_DOUBLE_2ADDR = 0xce,
                OP_REM_DOUBLE_2ADDR = 0xcf,
                OP_ADD_INT_LIT16 = 0xd0,
                OP_RSUB_INT = 0xd1,
                OP_MUL_INT_LIT16 = 0xd2,
                OP_DIV_INT_LIT16 = 0xd3,
                OP_REM_INT_LIT16 = 0xd4,
                OP_AND_INT_LIT16 = 0xd5,
                OP_OR_INT_LIT16 = 0xd6,
                OP_XOR_INT_LIT16 = 0xd7,
                OP_ADD_INT_LIT8 = 0xd8,
                OP_RSUB_INT_LIT8 = 0xd9,
                OP_MUL_INT_LIT8 = 0xda,
                OP_DIV_INT_LIT8 = 0xdb,
                OP_REM_INT_LIT8 = 0xdc,
                OP_AND_INT_LIT8 = 0xdd,
                OP_OR_INT_LIT8 = 0xde,
                OP_XOR_INT_LIT8 = 0xdf,
                OP_SHL_INT_LIT8 = 0xe0,
                OP_SHR_INT_LIT8 = 0xe1,
                OP_USHR_INT_LIT8 = 0xe2,
                OP_IGET_VOLATILE = 0xe3,
                OP_IPUT_VOLATILE = 0xe4,
                OP_SGET_VOLATILE = 0xe5,
                OP_SPUT_VOLATILE = 0xe6,
                OP_IGET_OBJECT_VOLATILE = 0xe7,
                OP_IGET_WIDE_VOLATILE = 0xe8,
                OP_IPUT_WIDE_VOLATILE = 0xe9,
                OP_SGET_WIDE_VOLATILE = 0xea,
                OP_SPUT_WIDE_VOLATILE = 0xeb,
                OP_BREAKPOINT = 0xec,
                OP_THROW_VERIFICATION_ERROR = 0xed,
                OP_EXECUTE_INLINE = 0xee,
                OP_EXECUTE_INLINE_RANGE = 0xef,
                OP_INVOKE_OBJECT_INIT_RANGE = 0xf0,
                OP_RETURN_VOID_BARRIER = 0xf1,
                OP_IGET_QUICK = 0xf2,
                OP_IGET_WIDE_QUICK = 0xf3,
                OP_IGET_OBJECT_QUICK = 0xf4,
                OP_IPUT_QUICK = 0xf5,
                OP_IPUT_WIDE_QUICK = 0xf6,
                OP_IPUT_OBJECT_QUICK = 0xf7,
                OP_INVOKE_VIRTUAL_QUICK = 0xf8,
                OP_INVOKE_VIRTUAL_QUICK_RANGE = 0xf9,
                OP_INVOKE_SUPER_QUICK = 0xfa,
                OP_INVOKE_SUPER_QUICK_RANGE = 0xfb,
                OP_IPUT_OBJECT_VOLATILE = 0xfc,
                OP_SGET_OBJECT_VOLATILE = 0xfd,
                OP_SPUT_OBJECT_VOLATILE = 0xfe,
                OP_CONST_METHOD_TYPE = 0xff,
                // END(libdex-opcode-enum)
                OP_PACKED_SWITCH_TABLE = 0x0100,
                OP_SPARSE_SWITCH_TABLE = 0x0200,
            };
        };

        static std::map<DVMTypes::Kind, std::string> KindString = {
            /**
             * @brief string representation for all the Kind enum values.
             */
            {DVMTypes::Kind::METH, "METH"},
            {DVMTypes::Kind::STRING, "STRING"},
            {DVMTypes::Kind::FIELD, "FIELD"},
            {DVMTypes::Kind::TYPE, "TYPE"},
            {DVMTypes::Kind::PROTO, "PROTO"},
            {DVMTypes::Kind::METH_PROTO, "METH_PROTO"},
            {DVMTypes::Kind::CALL_SITE, "CALL_SITE"},
            {DVMTypes::Kind::VARIES, "VARIES"},
            {DVMTypes::Kind::INLINE_METHOD, "INLINE_METHOD"},
            {DVMTypes::Kind::VTABLE_OFFSET, "VTABLE_OFFSET"},
            {DVMTypes::Kind::FIELD_OFFSET, "FIELD_OFFSET"},
            {DVMTypes::Kind::RAW_STRING, "RAW_STRING"},
            {DVMTypes::Kind::NONE_KIND, "NONE"}};

        static std::map<DVMTypes::ACCESS_FLAGS, std::string> ACCESS_FLAGS_STR = {
            /**
             * @brief string representation for the access flags.
             */
            {DVMTypes::ACCESS_FLAGS::ACC_PUBLIC, "public"},
            {DVMTypes::ACCESS_FLAGS::ACC_PRIVATE, "private"},
            {DVMTypes::ACCESS_FLAGS::ACC_PROTECTED, "protected"},
            {DVMTypes::ACCESS_FLAGS::ACC_STATIC, "static"},
            {DVMTypes::ACCESS_FLAGS::ACC_FINAL, "final"},
            {DVMTypes::ACCESS_FLAGS::ACC_SYNCHRONIZED, "synchronized"},
            {DVMTypes::ACCESS_FLAGS::ACC_BRIDGE, "bridge"},
            {DVMTypes::ACCESS_FLAGS::ACC_VARARGS, "varargs"},
            {DVMTypes::ACCESS_FLAGS::ACC_NATIVE, "native"},
            {DVMTypes::ACCESS_FLAGS::ACC_INTERFACE, "interface"},
            {DVMTypes::ACCESS_FLAGS::ACC_ABSTRACT, "abstract"},
            {DVMTypes::ACCESS_FLAGS::ACC_STRICT, "strictfp"},
            {DVMTypes::ACCESS_FLAGS::ACC_SYNTHETIC, "synthetic"},
            {DVMTypes::ACCESS_FLAGS::ACC_ENUM, "enum"},
            {DVMTypes::ACCESS_FLAGS::UNUSED, "unused"},
            {DVMTypes::ACCESS_FLAGS::ACC_CONSTRUCTOR, "constructor"},
            {DVMTypes::ACCESS_FLAGS::ACC_DECLARED_SYNCHRONIZED, "synchronized"}};

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