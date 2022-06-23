/***
 * File: dex_dalvik_opcodes.hpp
 * Author: @Farenain
 *
 * All the dalvik opcodes with information
 * of them, also strings will be used.
 *
 * Based on dvm.py from Androguard.
 */

#ifndef DEX_DALVIK_OPCODES_HPP
#define DEX_DALVIK_OPCODES_HPP

#include <iostream>
#include <map>

#include "KUNAI/DEX/DVM/dex_dvm_types.hpp"
#include "KUNAI/DEX/parser/dex_parser.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DalvikOpcodes;

        /**
         * @brief A shared_ptr of DalvikOpcodes a utility class to get
         * different values from the opcodes of the Dalvik machine these
         * can be strings, values, types, etc.
         */
        using dalvikopcodes_t = std::shared_ptr<DalvikOpcodes>;

        class DalvikOpcodes
        {
        public:
            /**
             * @brief DalvikOpcodes constructor.
             * @param dex_parser: dexparser_t object to use in the object.
             */
            DalvikOpcodes(dexparser_t dex_parser);

            /**
             * @brief DalvikOpcodes destructor.
             */
            ~DalvikOpcodes() = default;

            /**
             * @brief find the instruction opcode in the map to get instruction name.
             * @param instruction: std::uint32_t instruction opcode.
             * @return std::string
             */
            std::string get_instruction_name(std::uint32_t instruction);

            /**
             * @brief find the instruction Kind given an instruction opcode.
             * @param instruction: std::uint32_t instruction opcode.
             * @return DVMTypes::Kind
             */
            DVMTypes::Kind get_instruction_type(std::uint32_t instruction);

            /**
             * @brief find the instruction Operation given an instruction opcode.
             * @param instruction: std::uint32_t instruction opcode.
             * @return DVMTypes::Operation
             */
            DVMTypes::Operation get_instruction_operation(std::uint32_t instruction);

            /**
             * @brief get instruction type as string.
             * @param instruction: std::uint32_t instruction opcode.
             * @return std::string
             */
            std::string &get_instruction_type_str(std::uint32_t instruction);

            /**
             * @brief get string* by id.
             * @param id: std::uint32_t id of string.
             * @return std::string*
             */
            std::string *get_dalvik_string_by_id(std::uint32_t id)
            {
                return dex_parser->get_strings()->get_string_from_order(id);
            }

            /**
             * @brief get type_t by id.
             * @param id: std::uint32_t id of the Type.
             * @return type_t
             */
            type_t get_dalvik_Type_by_id(std::uint32_t id)
            {
                return dex_parser->get_types()->get_type_from_order(id);
            }

            /**
             * @brief get fieldid_t by id.
             * @param id: std::uint32_t of the FieldID.
             * @return fieldid_t
             */
            fieldid_t get_dalvik_field_by_id(std::uint32_t id)
            {
                return dex_parser->get_fields()->get_field_id_by_order(id);
            }

            /**
             * @brief get methodid_t by id.
             * @param id: std::uint32_t of the MethodID.
             * @return methodid_t
             */
            methodid_t get_dalvik_method_by_id(std::uint32_t id)
            {
                return dex_parser->get_methods()->get_method_by_order(id);
            }

            /**
             * @brief get protoid_t by id.
             * @param id: std::uint32_t of the ProtoID.
             * @return protoid_t
             */
            protoid_t get_dalvik_proto_by_id(std::uint32_t id)
            {
                return dex_parser->get_protos()->get_proto_by_order(id);
            }

            /**
             * @brief Get a dalvil EncodedField by a given fieldid_t
             * @param field: fieldid_t field to obtain its encodedfield_t
             * @return encodedfield_t
             */
            encodedfield_t get_dalvik_encoded_field_by_fieldid(fieldid_t field);

            /**
             * @brief get string by id.
             * @param id: std::uint32_t id of the string.
             * @return std::string
             */
            std::string &get_dalvik_string_by_id_str(std::uint32_t id)
            {
                return *dex_parser->get_strings()->get_string_from_order(id);
            }

            /**
             * @brief get raw string from Type by id.
             * @param id: std::uint32_t id of the Type.
             * @return std::string
             */
            std::string &get_dalvik_type_by_id_str(std::uint32_t id)
            {
                return dex_parser->get_types()->get_type_from_order(id)->get_raw();
            }

            /**
             * @brief get FieldID as string by id.
             * @param id: std::uint32_t id of the FieldID.
             * @return std::string
             */
            std::string get_dalvik_static_field_by_id_str(std::uint32_t id);

            /**
             * @brief get MethodID as string by id.
             * @param id: std::uint32_t id of the MethodID.
             * @return std::string
             */
            std::string get_dalvik_method_by_id_str(std::uint32_t id);

            /**
             * @brief get ProtoID as string by id.
             * @param id: std::uint32_t id of the ProtoID.
             * @return std::string
             */
            std::string get_dalvik_proto_by_id_str(std::uint32_t id);

            /**
             * @brief Method to get a string from the access flags.
             * @param acc_flag: value from enum DVMTypes::ACCESS_FLAGS.
             * @return std::string
             */
            std::string get_access_flags_string(DVMTypes::ACCESS_FLAGS acc_flag);

        private:
            dexparser_t dex_parser;

            /**
             * @brief Translation table from opcode to string
             */
            std::map<std::uint32_t, std::string> opcodes_instruction_name = {
                {DVMTypes::Opcode::OP_NOP, "nop"},
                {DVMTypes::Opcode::OP_MOVE, "move"},
                {DVMTypes::Opcode::OP_MOVE_FROM16, "move/from16"},
                {DVMTypes::Opcode::OP_MOVE_16, "move/16"},
                {DVMTypes::Opcode::OP_MOVE_WIDE, "move-wide"},
                {DVMTypes::Opcode::OP_MOVE_WIDE_FROM16, "move-wide/from16"},
                {DVMTypes::Opcode::OP_MOVE_WIDE_16, "move-wide/16"},
                {DVMTypes::Opcode::OP_MOVE_OBJECT, "move-object"},
                {DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16, "move-object/from16"},
                {DVMTypes::Opcode::OP_MOVE_OBJECT_16, "move-object/16"},
                {DVMTypes::Opcode::OP_MOVE_RESULT, "move-result"},
                {DVMTypes::Opcode::OP_MOVE_RESULT_WIDE, "move-result-wide"},
                {DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT, "move-result-object"},
                {DVMTypes::Opcode::OP_MOVE_EXCEPTION, "move-exception"},
                {DVMTypes::Opcode::OP_RETURN_VOID, "return-void"},
                {DVMTypes::Opcode::OP_RETURN, "return"},
                {DVMTypes::Opcode::OP_RETURN_WIDE, "return-wide"},
                {DVMTypes::Opcode::OP_RETURN_OBJECT, "return-object"},
                {DVMTypes::Opcode::OP_CONST_4, "const/4"},
                {DVMTypes::Opcode::OP_CONST_16, "const/16"},
                {DVMTypes::Opcode::OP_CONST, "const"},
                {DVMTypes::Opcode::OP_CONST_HIGH16, "const/high16"},
                {DVMTypes::Opcode::OP_CONST_WIDE_16, "const-wide/16"},
                {DVMTypes::Opcode::OP_CONST_WIDE_32, "const-wide/32"},
                {DVMTypes::Opcode::OP_CONST_WIDE, "const-wide"},
                {DVMTypes::Opcode::OP_CONST_WIDE_HIGH16, "const-wide/high16"},
                {DVMTypes::Opcode::OP_CONST_STRING, "const-string"},
                {DVMTypes::Opcode::OP_CONST_STRING_JUMBO, "const-string/jumbo"},
                {DVMTypes::Opcode::OP_CONST_CLASS, "const-class"},
                {DVMTypes::Opcode::OP_MONITOR_ENTER, "monitor-enter"},
                {DVMTypes::Opcode::OP_MONITOR_EXIT, "monitor-exit"},
                {DVMTypes::Opcode::OP_CHECK_CAST, "check-cast"},
                {DVMTypes::Opcode::OP_INSTANCE_OF, "instance-of"},
                {DVMTypes::Opcode::OP_ARRAY_LENGTH, "array-length"},
                {DVMTypes::Opcode::OP_NEW_INSTANCE, "new-instance"},
                {DVMTypes::Opcode::OP_NEW_ARRAY, "new-array"},
                {DVMTypes::Opcode::OP_FILLED_NEW_ARRAY, "filled-new-array"},
                {DVMTypes::Opcode::OP_FILLED_NEW_ARRAY_RANGE, "filled-new-array/range"},
                {DVMTypes::Opcode::OP_FILL_ARRAY_DATA, "fill-array-data"},
                {DVMTypes::Opcode::OP_THROW, "throw"},
                {DVMTypes::Opcode::OP_GOTO, "goto"},
                {DVMTypes::Opcode::OP_GOTO_16, "goto/16"},
                {DVMTypes::Opcode::OP_GOTO_32, "goto/32"},
                {DVMTypes::Opcode::OP_PACKED_SWITCH, "packed-switch"},
                {DVMTypes::Opcode::OP_SPARSE_SWITCH, "sparse-switch"},
                {DVMTypes::Opcode::OP_CMPL_FLOAT, "cmpl-float"},
                {DVMTypes::Opcode::OP_CMPG_FLOAT, "cmpg-float"},
                {DVMTypes::Opcode::OP_CMPL_DOUBLE, "cmpl-double"},
                {DVMTypes::Opcode::OP_CMPG_DOUBLE, "cmpg-double"},
                {DVMTypes::Opcode::OP_CMP_LONG, "cmp-long"},
                {DVMTypes::Opcode::OP_IF_EQ, "if-eq"},
                {DVMTypes::Opcode::OP_IF_NE, "if-ne"},
                {DVMTypes::Opcode::OP_IF_LT, "if-lt"},
                {DVMTypes::Opcode::OP_IF_GE, "if-ge"},
                {DVMTypes::Opcode::OP_IF_GT, "if-gt"},
                {DVMTypes::Opcode::OP_IF_LE, "if-le"},
                {DVMTypes::Opcode::OP_IF_EQZ, "if-eqz"},
                {DVMTypes::Opcode::OP_IF_NEZ, "if-nez"},
                {DVMTypes::Opcode::OP_IF_LTZ, "if-ltz"},
                {DVMTypes::Opcode::OP_IF_GEZ, "if-gez"},
                {DVMTypes::Opcode::OP_IF_GTZ, "if-gtz"},
                {DVMTypes::Opcode::OP_IF_LEZ, "if-lez"},
                {DVMTypes::Opcode::OP_UNUSED_3E, "unused"},
                {DVMTypes::Opcode::OP_UNUSED_3F, "unused"},
                {DVMTypes::Opcode::OP_UNUSED_40, "unused"},
                {DVMTypes::Opcode::OP_UNUSED_41, "unused"},
                {DVMTypes::Opcode::OP_UNUSED_42, "unused"},
                {DVMTypes::Opcode::OP_UNUSED_43, "unused"},
                {DVMTypes::Opcode::OP_AGET, "aget"},
                {DVMTypes::Opcode::OP_AGET_WIDE, "aget-wide"},
                {DVMTypes::Opcode::OP_AGET_OBJECT, "aget-object"},
                {DVMTypes::Opcode::OP_AGET_BOOLEAN, "aget-boolean"},
                {DVMTypes::Opcode::OP_AGET_BYTE, "aget-byte"},
                {DVMTypes::Opcode::OP_AGET_CHAR, "aget-char"},
                {DVMTypes::Opcode::OP_AGET_SHORT, "aget-short"},
                {DVMTypes::Opcode::OP_APUT, "aput"},
                {DVMTypes::Opcode::OP_APUT_WIDE, "aput-wide"},
                {DVMTypes::Opcode::OP_APUT_OBJECT, "aput-object"},
                {DVMTypes::Opcode::OP_APUT_BOOLEAN, "aput-boolean"},
                {DVMTypes::Opcode::OP_APUT_BYTE, "aput-byte"},
                {DVMTypes::Opcode::OP_APUT_CHAR, "aput-char"},
                {DVMTypes::Opcode::OP_APUT_SHORT, "aput-short"},
                {DVMTypes::Opcode::OP_IGET, "iget"},
                {DVMTypes::Opcode::OP_IGET_WIDE, "iget-wide"},
                {DVMTypes::Opcode::OP_IGET_OBJECT, "iget-object"},
                {DVMTypes::Opcode::OP_IGET_BOOLEAN, "iget-boolean"},
                {DVMTypes::Opcode::OP_IGET_BYTE, "iget-byte"},
                {DVMTypes::Opcode::OP_IGET_CHAR, "iget-char"},
                {DVMTypes::Opcode::OP_IGET_SHORT, "iget-short"},
                {DVMTypes::Opcode::OP_IPUT, "iput"},
                {DVMTypes::Opcode::OP_IPUT_WIDE, "iput-wide"},
                {DVMTypes::Opcode::OP_IPUT_OBJECT, "iput-object"},
                {DVMTypes::Opcode::OP_IPUT_BOOLEAN, "iput-boolean"},
                {DVMTypes::Opcode::OP_IPUT_BYTE, "iput-byte"},
                {DVMTypes::Opcode::OP_IPUT_CHAR, "iput-char"},
                {DVMTypes::Opcode::OP_IPUT_SHORT, "iput-short"},
                {DVMTypes::Opcode::OP_SGET, "sget"},
                {DVMTypes::Opcode::OP_SGET_WIDE, "sget-wide"},
                {DVMTypes::Opcode::OP_SGET_OBJECT, "sget-object"},
                {DVMTypes::Opcode::OP_SGET_BOOLEAN, "sget-boolean"},
                {DVMTypes::Opcode::OP_SGET_BYTE, "sget-byte"},
                {DVMTypes::Opcode::OP_SGET_CHAR, "sget-char"},
                {DVMTypes::Opcode::OP_SGET_SHORT, "sget-short"},
                {DVMTypes::Opcode::OP_SPUT, "sput"},
                {DVMTypes::Opcode::OP_SPUT_WIDE, "sput-wide"},
                {DVMTypes::Opcode::OP_SPUT_OBJECT, "sput-object"},
                {DVMTypes::Opcode::OP_SPUT_BOOLEAN, "sput-boolean"},
                {DVMTypes::Opcode::OP_SPUT_BYTE, "sput-byte"},
                {DVMTypes::Opcode::OP_SPUT_CHAR, "sput-char"},
                {DVMTypes::Opcode::OP_SPUT_SHORT, "sput-short"},
                {DVMTypes::Opcode::OP_INVOKE_VIRTUAL, "invoke-virtual"},
                {DVMTypes::Opcode::OP_INVOKE_SUPER, "invoke-super"},
                {DVMTypes::Opcode::OP_INVOKE_DIRECT, "invoke-direct"},
                {DVMTypes::Opcode::OP_INVOKE_STATIC, "invoke-static"},
                {DVMTypes::Opcode::OP_INVOKE_INTERFACE, "invoke-interface"},
                {DVMTypes::Opcode::OP_UNUSED_73, "unused"},
                {DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE, "invoke-virtual/range"},
                {DVMTypes::Opcode::OP_INVOKE_SUPER_RANGE, "invoke-super/range"},
                {DVMTypes::Opcode::OP_INVOKE_DIRECT_RANGE, "invoke-direct/range"},
                {DVMTypes::Opcode::OP_INVOKE_STATIC_RANGE, "invoke-static/range"},
                {DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE, "invoke-interface/range"},
                {DVMTypes::Opcode::OP_UNUSED_79, "unused"},
                {DVMTypes::Opcode::OP_UNUSED_7A, "unused"},
                {DVMTypes::Opcode::OP_NEG_INT, "neg-int"},
                {DVMTypes::Opcode::OP_NOT_INT, "not-int"},
                {DVMTypes::Opcode::OP_NEG_LONG, "neg-long"},
                {DVMTypes::Opcode::OP_NOT_LONG, "not-long"},
                {DVMTypes::Opcode::OP_NEG_FLOAT, "neg-float"},
                {DVMTypes::Opcode::OP_NEG_DOUBLE, "neg-double"},
                {DVMTypes::Opcode::OP_INT_TO_LONG, "int-to-long"},
                {DVMTypes::Opcode::OP_INT_TO_FLOAT, "int-to-float"},
                {DVMTypes::Opcode::OP_INT_TO_DOUBLE, "int-to-double"},
                {DVMTypes::Opcode::OP_LONG_TO_INT, "long-to-int"},
                {DVMTypes::Opcode::OP_LONG_TO_FLOAT, "long-to-float"},
                {DVMTypes::Opcode::OP_LONG_TO_DOUBLE, "long-to-double"},
                {DVMTypes::Opcode::OP_FLOAT_TO_INT, "float-to-int"},
                {DVMTypes::Opcode::OP_FLOAT_TO_LONG, "float-to-long"},
                {DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE, "float-to-double"},
                {DVMTypes::Opcode::OP_DOUBLE_TO_INT, "double-to-int"},
                {DVMTypes::Opcode::OP_DOUBLE_TO_LONG, "double-to-long"},
                {DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT, "double-to-float"},
                {DVMTypes::Opcode::OP_INT_TO_BYTE, "int-to-byte"},
                {DVMTypes::Opcode::OP_INT_TO_CHAR, "int-to-char"},
                {DVMTypes::Opcode::OP_INT_TO_SHORT, "int-to-short"},
                {DVMTypes::Opcode::OP_ADD_INT, "add-int"},
                {DVMTypes::Opcode::OP_SUB_INT, "sub-int"},
                {DVMTypes::Opcode::OP_MUL_INT, "mul-int"},
                {DVMTypes::Opcode::OP_DIV_INT, "div-int"},
                {DVMTypes::Opcode::OP_REM_INT, "rem-int"},
                {DVMTypes::Opcode::OP_AND_INT, "and-int"},
                {DVMTypes::Opcode::OP_OR_INT, "or-int"},
                {DVMTypes::Opcode::OP_XOR_INT, "xor-int"},
                {DVMTypes::Opcode::OP_SHL_INT, "shl-int"},
                {DVMTypes::Opcode::OP_SHR_INT, "shr-int"},
                {DVMTypes::Opcode::OP_USHR_INT, "ushr-int"},
                {DVMTypes::Opcode::OP_ADD_LONG, "add-long"},
                {DVMTypes::Opcode::OP_SUB_LONG, "sub-long"},
                {DVMTypes::Opcode::OP_MUL_LONG, "mul-long"},
                {DVMTypes::Opcode::OP_DIV_LONG, "div-long"},
                {DVMTypes::Opcode::OP_REM_LONG, "rem-long"},
                {DVMTypes::Opcode::OP_AND_LONG, "and-long"},
                {DVMTypes::Opcode::OP_OR_LONG, "or-long"},
                {DVMTypes::Opcode::OP_XOR_LONG, "xor-long"},
                {DVMTypes::Opcode::OP_SHL_LONG, "shl-long"},
                {DVMTypes::Opcode::OP_SHR_LONG, "shr-long"},
                {DVMTypes::Opcode::OP_USHR_LONG, "ushr-long"},
                {DVMTypes::Opcode::OP_ADD_FLOAT, "add-float"},
                {DVMTypes::Opcode::OP_SUB_FLOAT, "sub-float"},
                {DVMTypes::Opcode::OP_MUL_FLOAT, "mul-float"},
                {DVMTypes::Opcode::OP_DIV_FLOAT, "div-float"},
                {DVMTypes::Opcode::OP_REM_FLOAT, "rem-float"},
                {DVMTypes::Opcode::OP_ADD_DOUBLE, "add-double"},
                {DVMTypes::Opcode::OP_SUB_DOUBLE, "sub-double"},
                {DVMTypes::Opcode::OP_MUL_DOUBLE, "mul-double"},
                {DVMTypes::Opcode::OP_DIV_DOUBLE, "div-double"},
                {DVMTypes::Opcode::OP_REM_DOUBLE, "rem-double"},
                {DVMTypes::Opcode::OP_ADD_INT_2ADDR, "add-int/2addr"},
                {DVMTypes::Opcode::OP_SUB_INT_2ADDR, "sub-int/2addr"},
                {DVMTypes::Opcode::OP_MUL_INT_2ADDR, "mul-int/2addr"},
                {DVMTypes::Opcode::OP_DIV_INT_2ADDR, "div-int/2addr"},
                {DVMTypes::Opcode::OP_REM_INT_2ADDR, "rem-int/2addr"},
                {DVMTypes::Opcode::OP_AND_INT_2ADDR, "and-int/2addr"},
                {DVMTypes::Opcode::OP_OR_INT_2ADDR, "or-int/2addr"},
                {DVMTypes::Opcode::OP_XOR_INT_2ADDR, "xor-int/2addr"},
                {DVMTypes::Opcode::OP_SHL_INT_2ADDR, "shl-int/2addr"},
                {DVMTypes::Opcode::OP_SHR_INT_2ADDR, "shr-int/2addr"},
                {DVMTypes::Opcode::OP_USHR_INT_2ADDR, "ushr-int/2addr"},
                {DVMTypes::Opcode::OP_ADD_LONG_2ADDR, "add-long/2addr"},
                {DVMTypes::Opcode::OP_SUB_LONG_2ADDR, "sub-long/2addr"},
                {DVMTypes::Opcode::OP_MUL_LONG_2ADDR, "mul-long/2addr"},
                {DVMTypes::Opcode::OP_DIV_LONG_2ADDR, "div-long/2addr"},
                {DVMTypes::Opcode::OP_REM_LONG_2ADDR, "rem-long/2addr"},
                {DVMTypes::Opcode::OP_AND_LONG_2ADDR, "and-long/2addr"},
                {DVMTypes::Opcode::OP_OR_LONG_2ADDR, "or-long/2addr"},
                {DVMTypes::Opcode::OP_XOR_LONG_2ADDR, "xor-long/2addr"},
                {DVMTypes::Opcode::OP_SHL_LONG_2ADDR, "shl-long/2addr"},
                {DVMTypes::Opcode::OP_SHR_LONG_2ADDR, "shr-long/2addr"},
                {DVMTypes::Opcode::OP_USHR_LONG_2ADDR, "ushr-long/2addr"},
                {DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR, "add-float/2addr"},
                {DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR, "sub-float/2addr"},
                {DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR, "mul-float/2addr"},
                {DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR, "div-float/2addr"},
                {DVMTypes::Opcode::OP_REM_FLOAT_2ADDR, "rem-float/2addr"},
                {DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR, "add-double/2addr"},
                {DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR, "sub-double/2addr"},
                {DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR, "mul-double/2addr"},
                {DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR, "div-double/2addr"},
                {DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR, "rem-double/2addr"},
                {DVMTypes::Opcode::OP_ADD_INT_LIT16, "add-int/lit16"},
                {DVMTypes::Opcode::OP_RSUB_INT, "rsub-int"},
                {DVMTypes::Opcode::OP_MUL_INT_LIT16, "mul-int/lit16"},
                {DVMTypes::Opcode::OP_DIV_INT_LIT16, "div-int/lit16"},
                {DVMTypes::Opcode::OP_REM_INT_LIT16, "rem-int/lit16"},
                {DVMTypes::Opcode::OP_AND_INT_LIT16, "and-int/lit16"},
                {DVMTypes::Opcode::OP_OR_INT_LIT16, "or-int/lit16"},
                {DVMTypes::Opcode::OP_XOR_INT_LIT16, "xor-int/lit16"},
                {DVMTypes::Opcode::OP_ADD_INT_LIT8, "add-int/lit8"},
                {DVMTypes::Opcode::OP_RSUB_INT_LIT8, "rsub-int/lit8"},
                {DVMTypes::Opcode::OP_MUL_INT_LIT8, "mul-int/lit8"},
                {DVMTypes::Opcode::OP_DIV_INT_LIT8, "div-int/lit8"},
                {DVMTypes::Opcode::OP_REM_INT_LIT8, "rem-int/lit8"},
                {DVMTypes::Opcode::OP_AND_INT_LIT8, "and-int/lit8"},
                {DVMTypes::Opcode::OP_OR_INT_LIT8, "or-int/lit8"},
                {DVMTypes::Opcode::OP_XOR_INT_LIT8, "xor-int/lit8"},
                {DVMTypes::Opcode::OP_SHL_INT_LIT8, "shl-int/lit8"},
                {DVMTypes::Opcode::OP_SHR_INT_LIT8, "shr-int/lit8"},
                {DVMTypes::Opcode::OP_USHR_INT_LIT8, "ushr-int/lit8"},
                {DVMTypes::Opcode::OP_IGET_VOLATILE, "iget-volatile"},
                {DVMTypes::Opcode::OP_IPUT_VOLATILE, "iput-volatile"},
                {DVMTypes::Opcode::OP_SGET_VOLATILE, "sget-volatile"},
                {DVMTypes::Opcode::OP_SPUT_VOLATILE, "sput-volatile"},
                {DVMTypes::Opcode::OP_IGET_OBJECT_VOLATILE, "iget-object-volatile"},
                {DVMTypes::Opcode::OP_IGET_WIDE_VOLATILE, "iget-wide-volatile"},
                {DVMTypes::Opcode::OP_IPUT_WIDE_VOLATILE, "iput-wide-volatile"},
                {DVMTypes::Opcode::OP_SGET_WIDE_VOLATILE, "sget-wide-volatile"},
                {DVMTypes::Opcode::OP_SPUT_WIDE_VOLATILE, "sput-wide-volatile"},
                {DVMTypes::Opcode::OP_BREAKPOINT, "breakpoint"},
                {DVMTypes::Opcode::OP_THROW_VERIFICATION_ERROR, "throw-verification-error"},
                {DVMTypes::Opcode::OP_EXECUTE_INLINE, "execute-inline"},
                {DVMTypes::Opcode::OP_EXECUTE_INLINE_RANGE, "execute-inline-range"},
                {DVMTypes::Opcode::OP_INVOKE_OBJECT_INIT_RANGE, "invoke-object-init-range"},
                {DVMTypes::Opcode::OP_RETURN_VOID_BARRIER, "return-void-barrier"},
                {DVMTypes::Opcode::OP_IGET_QUICK, "iget-quick"},
                {DVMTypes::Opcode::OP_IGET_WIDE_QUICK, "iget-wide-quick"},
                {DVMTypes::Opcode::OP_IGET_OBJECT_QUICK, "iget-object-quick"},
                {DVMTypes::Opcode::OP_IPUT_QUICK, "iput-quick"},
                {DVMTypes::Opcode::OP_IPUT_WIDE_QUICK, "iput-wide-quick"},
                {DVMTypes::Opcode::OP_IPUT_OBJECT_QUICK, "iput-object-quick"},
                {DVMTypes::Opcode::OP_INVOKE_VIRTUAL_QUICK, "invoke-virtual-quick"},
                {DVMTypes::Opcode::OP_INVOKE_VIRTUAL_QUICK_RANGE, "invoke-virtual-quick-range"},
                {DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK, "invoke-polymorphic"},
                {DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK_RANGE, "invoke-polymorphic/range"},
                {DVMTypes::Opcode::OP_IPUT_OBJECT_VOLATILE, "invoke-custom"},
                {DVMTypes::Opcode::OP_SGET_OBJECT_VOLATILE, "invoke-custom/range"},
                {DVMTypes::Opcode::OP_SPUT_OBJECT_VOLATILE, "const-method-handle"},
                {DVMTypes::Opcode::OP_CONST_METHOD_TYPE, "const-method-type"},
                // special cases...
                {0x0100, "packed-switch-payload"},
                {0x0200, "sparse-switch-payload"},
                {0x0300, "fill-array-data-payload"},
                {0xf2ff, "invoke-object-init/jumbo"},
                {0xf3ff, "iget-volatile/jumbo"},
                {0xf4ff, "iget-wide-volatile/jumbo"},
                {0xf5ff, "iget-object-volatile/jumbo"},
                {0xf6ff, "iput-volatile/jumbo"},
                {0xf7ff, "iput-wide-volatile/jumbo"},
                {0xf8ff, "iput-object-volatile/jumbo"},
                {0xf9ff, "sget-volatile/jumbo"},
                {0xfaff, "sget-wide-volatile/jumbo"},
                {0xfbff, "sget-object-volatile/jumbo"},
                {0xfcff, "sput-volatile/jumbo"},
                {0xfdff, "sput-wide-volatile/jumbo"},
                {0xfeff, "sput-object-volatile/jumbo"},
                {0xffff, "throw-verification-error/jumbo"}};

            /**
             * @brief Translation table from opcode to instruction type.
             */
            std::map<std::uint32_t, DVMTypes::Kind> opcodes_instruction_type = {
                {DVMTypes::Opcode::OP_CONST_STRING, DVMTypes::Kind::STRING},
                {DVMTypes::Opcode::OP_CONST_STRING_JUMBO, DVMTypes::Kind::STRING},
                {DVMTypes::Opcode::OP_CONST_CLASS, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_CHECK_CAST, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_INSTANCE_OF, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_NEW_INSTANCE, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_NEW_ARRAY, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_FILLED_NEW_ARRAY, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_FILLED_NEW_ARRAY_RANGE, DVMTypes::Kind::TYPE},
                {DVMTypes::Opcode::OP_IGET, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IGET_WIDE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IGET_OBJECT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IGET_BOOLEAN, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IGET_BYTE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IGET_CHAR, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IGET_SHORT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT_WIDE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT_OBJECT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT_BOOLEAN, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT_BYTE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT_CHAR, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_IPUT_SHORT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET_WIDE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET_OBJECT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET_BOOLEAN, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET_BYTE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET_CHAR, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SGET_SHORT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT_WIDE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT_OBJECT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT_BOOLEAN, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT_BYTE, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT_CHAR, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_SPUT_SHORT, DVMTypes::Kind::FIELD},
                {DVMTypes::Opcode::OP_INVOKE_VIRTUAL, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_SUPER, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_DIRECT, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_STATIC, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_INTERFACE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_SUPER_RANGE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_DIRECT_RANGE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_STATIC_RANGE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK, DVMTypes::Kind::METH_PROTO},
                {DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK_RANGE, DVMTypes::Kind::METH_PROTO},
                {DVMTypes::Opcode::OP_IPUT_OBJECT_VOLATILE, DVMTypes::Kind::CALL_SITE},
                {DVMTypes::Opcode::OP_SGET_OBJECT_VOLATILE, DVMTypes::Kind::CALL_SITE},
                {DVMTypes::Opcode::OP_SPUT_OBJECT_VOLATILE, DVMTypes::Kind::METH},
                {DVMTypes::Opcode::OP_CONST_METHOD_TYPE, DVMTypes::Kind::PROTO},
                // special cases
                {0x0100, DVMTypes::Kind::NONE_KIND},
                {0x0200, DVMTypes::Kind::NONE_KIND},
                {0x0300, DVMTypes::Kind::NONE_KIND},
                {0xf2ff, DVMTypes::Kind::METH},
                {0xf3ff, DVMTypes::Kind::FIELD},
                {0xf4ff, DVMTypes::Kind::FIELD},
                {0xf5ff, DVMTypes::Kind::FIELD},
                {0xf6ff, DVMTypes::Kind::FIELD},
                {0xf7ff, DVMTypes::Kind::FIELD},
                {0xf8ff, DVMTypes::Kind::FIELD},
                {0xf9ff, DVMTypes::Kind::FIELD},
                {0xfaff, DVMTypes::Kind::FIELD},
                {0xfbff, DVMTypes::Kind::FIELD},
                {0xfcff, DVMTypes::Kind::FIELD},
                {0xfdff, DVMTypes::Kind::FIELD},
                {0xfeff, DVMTypes::Kind::FIELD},
                {0xffff, DVMTypes::Kind::VARIES}};

            /**
             * @brief instructions that makes some specific operation
             *        branch, break, read, write...
             */
            std::map<std::uint32_t, DVMTypes::Operation> opcode_instruction_operation = {
                // branch instructions
                {0x27, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x32, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x33, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x34, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x35, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x36, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x37, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x38, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x39, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x3A, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x3B, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x3C, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x3D, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x28, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x29, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x2A, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x0E, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x0F, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x10, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x11, DVMTypes::Operation::BRANCH_DVM_OPCODE},
                {0x2B, DVMTypes::Operation::BRANCH_DVM_OPCODE}, // packed-switch
                {0x2C, DVMTypes::Operation::BRANCH_DVM_OPCODE}, // sparse-switch
                // break instruction
                {0x6E, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x6F, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x70, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x71, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x72, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x74, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x75, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x76, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x77, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x78, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x01, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x02, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x03, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x04, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x05, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x06, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x07, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x08, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x09, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x0A, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x0B, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x0C, DVMTypes::Operation::BREAK_DVM_OPCODE},
                {0x0D, DVMTypes::Operation::BREAK_DVM_OPCODE},
                // read field instruction
                {0x44, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x45, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x46, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x47, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x48, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x49, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x4A, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x52, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x53, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x54, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x55, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x56, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x57, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x58, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x60, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x61, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x62, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x63, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x64, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x65, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0x66, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0xf3ff, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0xf4ff, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0xf5ff, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0xf9ff, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0xfaff, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                {0xfbff, DVMTypes::Operation::FIELD_READ_DVM_OPCODE},
                // write field instruction
                {0x4B, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x4C, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x4D, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x4E, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x4F, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x50, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x51, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x59, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x5A, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x5B, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x5C, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x5D, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x5E, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x5F, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x67, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x68, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x69, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x6A, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x6B, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x6C, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0x6D, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0xf6ff, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0xf7ff, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0xf8ff, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0xfcff, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0xfdff, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
                {0xfeff, DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE},
            };
        };
    }
}

#endif