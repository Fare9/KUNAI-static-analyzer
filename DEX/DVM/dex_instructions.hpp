/***
 * File: dex_instructions.hpp
 * Author: @Farenain
 * 
 * All the instructions from the dalvik machine
 * represented as classes.
 */

#ifndef DEX_INSTRUCTIONS_HPP
#define DEX_INSTRUCTIONS_HPP

#include <any>
#include <iostream>
#include <memory>
#include <fstream>
#include <tuple>
#include <vector>
#include <iomanip>

#include "dex_dvm_types.hpp"
#include "dex_dalvik_opcodes.hpp"
#include "utils.hpp"
#include "exceptions.hpp"

namespace KUNAI
{
    namespace DEX
    {

        class Instruction
        {
        public:
            Instruction(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction();

            DVMTypes::Kind get_kind();
            std::string get_translated_kind();
            std::string get_name();
            std::uint32_t get_length();
            std::uint32_t get_OP();
            std::shared_ptr<DalvikOpcodes> get_dalvik_opcodes();

            void set_length(std::uint32_t length);
            void set_OP(std::uint32_t OP);

            virtual void show_instruction();
            virtual void give_me_instruction(std::ostream &os);
            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();
        
            void set_number_of_registers(std::uint32_t number_of_registers);
            void set_number_of_parameters(std::uint32_t number_of_parameters);
            std::uint32_t get_number_of_registers();
            std::uint32_t get_number_of_parameters();
            std::string get_register_correct_representation(std::uint32_t reg);
        private:
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::uint32_t length;
            std::uint32_t OP;

            std::uint32_t number_of_registers;
            std::uint32_t number_of_parameters;
        };

        class Instruction00x : public Instruction
        {
        public:
            Instruction00x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction00x();
        };

        class Instruction10x : public Instruction
        /***
         * Waste cycles. 
         * 
         * Example of instruction:
         * 
         * nop
         * 
         * length = 2 bytes
         */
        {
        public:
            Instruction10x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction10x();

            virtual std::uint64_t get_raw();
        };

        class Instruction12x : public Instruction
        /***
         * Move the contents of one non-object register to another.
         * 
         * Example of instruction:
         * 
         * move vA, vB
         * 
         * vA: dest reg (4 bits)
         * vB: src reg (4 bits)
         */
        {
        public:
            Instruction12x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction12x();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::uint8_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vA; // destination
            std::uint8_t vB; // source
        };

        class Instruction11n : public Instruction
        /***
         * Move the given literal value
         * 
         * Example Instruction:
         * 
         * const/4 vA, #+B
         * 
         * vA: destination register (4 bits)
         * #+B: signed int (4 bits)
         */
        {
        public:
            Instruction11n(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction11n();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::int8_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vA;
            std::int8_t nB;
        };

        class Instruction11x : public Instruction
        /***
         * Move single, double-words or object from invoke results, also
         * save caught exception into given register.
         * 
         * Example of instruction:
         * 
         * move-result vAA
         * 
         * vAA: destination register (8 bits)
         */
        {
        public:
            Instruction11x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction11x();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
        };

        class Instruction10t : public Instruction
        /***
         * Unconditionally jump to indicated instruction.
         * 
         * Example Instruction:
         *  goto +AA
         * 
         * +AA: signed branch offset, cannot be 0 (8 bits)
         */
        {
        public:
            Instruction10t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction10t();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_offset_type();
            std::int8_t get_offset();

        private:
            std::int8_t nAA;
        };

        class Instruction20t : public Instruction
        /***
         * Unconditionally jump to indicated instruction
         * 
         * Example Instruction:
         *  goto/16 +AAAA
         *
         * +AAAA: signed branch offset, cannot be 0 (16 bits).
         */
        {
        public:
            Instruction20t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction20t();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_offset_type();
            std::int16_t get_offset();

        private:
            std::int16_t nAAAA;
        };

        class Instruction20bc : public Instruction
        /***
         * opAA, kind@BBBB
         * 
         * AA: type of error (8 bits)
         * BBBB: index into appropiate table (16 bits)
         */
        {
        public:
            Instruction20bc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction20bc();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_type_of_error_data_type();
            std::uint8_t get_type_of_error();

            DVMTypes::Operand get_index_table_data_type();
            std::uint16_t get_index_table();

        private:
            std::uint8_t nAA;
            std::uint16_t nBBBB;
        };

        class Instruction22x : public Instruction
        /***
         * Move the contents of one non-object register to another.
         * 
         * Example of instruction:
         * 
         * move/from16 vAA, vBBBB
         * 
         * vAA: destination register (8 bits)
         * vBBBB: source register (16 bits)
         */
        {
        public:
            Instruction22x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction22x();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::uint16_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
            std::uint16_t vBBBB;
        };

        class Instruction21t : public Instruction
        /***
         * Branch to the given destination if the given register's value 
         * compares with 0 as specified.
         * 
         * Example Instruction:
         *  if-testz vAA, +BBBB
         * 
         * vAA: register to test (8 bits).
         * +BBBB: signed branch offset (16 bits).
         */
        {
        public:
            Instruction21t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction21t();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_check_reg_type();
            std::uint8_t get_check_reg();

            DVMTypes::Operand get_ref_type();
            std::int16_t get_ref();

        private:
            std::uint8_t vAA;
            std::int16_t nBBBB;
        };

        class Instruction21s : public Instruction
        /***
         * Move given literal value into specified register.
         * 
         * Example Instruction:
         * 
         * const/16 vAA, #+BBBB
         * 
         * vAA: destination register (8 bits)
         * #+BBBB: signed int (16 bits)
         */
        {
        public:
            Instruction21s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction21s();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::int16_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vA;
            std::int16_t nBBBB;
        };

        class Instruction21h : public Instruction
        /***
         * Move given literal value into specified register.
         * 
         * Example Instruction:
         * 
         * const/high16 vAA, #+BBBB0000
         * 
         * vAA: destination register (8 bits).
         * #+BBBB0000: signed int (16 bits)
         */
        {
        public:
            Instruction21h(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction21h();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::int64_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
            std::int64_t nBBBB;
            std::int16_t nBBBB_aux;
        };

        class Instruction21c : public Instruction
        /***
         * Move a reference to register from a string, type, etc.
         * 
         * Example instruction:
         * 
         * const-string vAA, string@BBBB
         * 
         * vAA: destination register (8 bits)
         * string/type@BBBB: string/type index (16 bits)
         */
        {
        public:
            Instruction21c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction21c();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::uint16_t get_source();

            DVMTypes::Kind get_source_kind();
            std::string *get_source_str();
            Type *get_source_typeid();
            FieldID *get_source_static_field();
            MethodID *get_source_method();
            ProtoID *get_source_proto();

            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
            std::uint16_t iBBBB;
        };

        class Instruction23x : public Instruction
        /***
         * Perform indicated floating point or long comparison.
         * 
         * Example Instruction:
         *  cmpkind vAA, vBB, vCC
         * 
         * vAA: destination register (8 bits).
         * vBB: first source register or pair (8 bits).
         * vCC: second source register or pair (8 bits).
         */
        {
        public:
            Instruction23x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction23x();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

            DVMTypes::Operand get_first_source_type();
            std::uint8_t get_first_source();

            DVMTypes::Operand get_second_source_type();
            std::uint8_t get_second_source();

        private:
            std::uint8_t vAA;
            std::uint8_t vBB;
            std::uint8_t vCC;
        };

        class Instruction22b : public Instruction
        /***
         * Perform the indicated binary op on the indicated register (first argument)
         * and literal value (second argument), storing the result in the destination register. 
         * 
         * Example Instruction:
         *  add-int/lit8 vAA, vBB, #+CC
         * 
         * vAA: destination register (8 bits).
         * vBB: source register (8 bits).
         * +CC: signed int constant (8 bits).
         */
        {
        public:
            Instruction22b(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction22b();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

            DVMTypes::Operand get_source_type();
            std::uint8_t get_source();

            DVMTypes::Operand get_number_type();
            std::int8_t get_number();

        private:
            std::uint8_t vAA;
            std::uint8_t vBB;
            std::int8_t nCC;
        };

        class Instruction22t : public Instruction
        /***
         * Branch to given destination if given two registers' values compare
         * as specified.
         * 
         * Example Instruction:
         *  if-test vA, vB, +CCCC
         * 
         * vA: first register to test (4 bits)
         * vB: second register to test (4 bits)
         * +CCCC: signed branch offset (16 bits)
         */
        {
        public:
            Instruction22t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction22t();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_first_check_reg_type();
            std::uint8_t get_first_check_reg();

            DVMTypes::Operand get_second_check_reg_type();
            std::uint8_t get_second_check_reg();

            DVMTypes::Operand get_ref_type();
            std::int16_t get_ref();

        private:
            std::uint8_t vA;
            std::uint8_t vB;
            std::int16_t nCCCC;
        };

        class Instruction22s : public Instruction
        /***
         * Perform the indicated binary op on the indicated register (first argument) 
         * and literal value (second argument), storing the result in the destination register. 
         * 
         * Example Instruction:
         *  add-int/lit16 vA, vB, #+CCCC
         * 
         * vA: destination register (4 bits)
         * vB: source register (4 bits)
         * +CCCC: signed int constant (16 bits)
         */
        {
        public:
            Instruction22s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction22s();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

            DVMTypes::Operand get_source_type();
            std::uint8_t get_source();

            DVMTypes::Operand get_number_type();
            std::int16_t get_number();

        private:
            std::uint8_t vA;
            std::uint8_t vB;
            std::uint16_t nCCCC;
        };

        class Instruction22c : public Instruction
        /***
         * Store in the given destination register 1 if the indicated 
         * reference is an instance of the given type/field, or 0 if not. 
         * 
         * Example Instruction:
         * 
         * instance-of vA, vB, type@CCCC
         * 
         * vA: destination register (4 bits).
         * vB: reference-bearing register (4 bits).
         * field/type@CCCC: field/type index (16 bits)
         */
        {
        public:
            Instruction22c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction22c();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_first_operand_type();
            std::uint8_t get_first_operand();

            DVMTypes::Operand get_second_operand_type();
            std::uint8_t get_second_operand();

            DVMTypes::Operand get_third_operand_type();
            std::uint16_t get_third_operand();
            DVMTypes::Kind get_third_operand_kind();
            Type *get_third_operand_typeId();
            FieldID *get_third_operand_FieldId();

        private:
            std::uint8_t vA;
            std::uint8_t vB;
            std::uint16_t iCCCC;
        };

        class Instruction22cs : public Instruction
        /***
         * suggested format for statically linked field 
         * access instructions of format 22c
         * 
         * op vA, vB, fieldoff@CCCC
         * 
         * vA: destination register (4 bits).
         * vB: reference-beraing register (4 bits).
         * fieldoff@CCCC: field/type index (16 bits)
         */
        {
        public:
            Instruction22cs(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction22cs();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_first_operand_type();
            std::uint8_t get_first_operand();

            DVMTypes::Operand get_second_operand_type();
            std::uint8_t get_second_operand();

            DVMTypes::Operand get_third_operand_type();
            std::uint16_t get_third_operand();
            DVMTypes::Kind get_third_operand_kind();
            Type *get_third_operand_typeId();
            FieldID *get_third_operand_FieldId();

        private:
            std::uint8_t vA;
            std::uint8_t vB;
            std::uint16_t iCCCC;
        };

        class Instruction30t : public Instruction
        /***
         * Unconditionally jump to the indicated instruction.
         * 
         * Example Instruction:
         *  goto/32 +AAAAAAAA
         * 
         * +AAAAAAAA: signed branch offset (32 bits).
         */
        {
        public:
            Instruction30t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction30t();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_offset_type();
            std::int32_t get_offset();

        private:
            std::int32_t nAAAAAAAA;
        };

        class Instruction32x : public Instruction
        /***
         * Move the contents of one non-object register to another.
         * 
         * Example of instruction:
         * 
         * move/16 vAAAA, vBBBB
         * 
         * vAAAA: destination register (16 bits)
         * vBBBB: source register (16 bits)
         */
        {
        public:
            Instruction32x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction32x();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::uint16_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint16_t get_destination();

        private:
            std::uint16_t vAAAA;
            std::uint16_t vBBBB;
        };

        class Instruction31i : public Instruction
        /***
         * Move given literal value into specified register
         * 
         * Example Instruction:
         * 
         * const vAA, #+BBBBBBBB
         * 
         * vAA: destination register (8 bits)
         * #+BBBBBBBB: arbitrary 32-bit constant
         */
        {
        public:
            Instruction31i(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction31i();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::int32_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
            std::uint32_t nBBBBBBBB;
        };

        class Instruction31t : public Instruction
        /***
         * Fill given array with indicated data. Reference must
         * be an array of primitives.
         * 
         * Example Instruction:
         *  fill-array-data vAA, +BBBBBBBB
         * 
         * vAA: array reference (8 bits)
         * +BBBBBBBB: signed "branch" offset to table data pseudo instruction (32 bits).
         */
        {
        public:
            Instruction31t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction31t();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_array_ref_type();
            std::uint8_t get_array_ref();

            DVMTypes::Operand get_offset_type();
            std::int32_t get_offset();

        private:
            std::uint8_t vAA;
            std::int32_t nBBBBBBBB;
        };

        class Instruction31c : public Instruction
        /***
         * Move a reference to the string specified by the given index into the specified register.
         * 
         * Example Instruction:
         * 
         * const-string/jumbo vAA, string@BBBBBBBB
         * 
         * vAA: destination register (8 bits).
         * string@BBBBBBBB: string index (32 bits)
         */
        {
        public:
            Instruction31c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction31c();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::uint16_t get_source();

            DVMTypes::Kind get_source_kind();
            std::string *get_source_str();

            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
            std::uint32_t iBBBBBBBB;
        };

        class Instruction35c : public Instruction
        /***
         * Construct array of given type and size, filling it with supplied
         * contents. Type must be an array type. Array's contents must be
         * single-word.
         * 
         * Example instruction:
         * 
         *  filled-new-array {vC, vD, vE, vF, vG}, type@BBBB
         * 
         * A: array size and argument word count (4 bits).
         * B: type index (16 bits).
         * C..G: argument registers (4 bits each).
         */
        {
        public:
            Instruction35c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction35c();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::uint8_t get_array_size();

            std::uint16_t get_type_index();

            DVMTypes::Operand get_operands_types();
            Type *get_operands_kind_type();
            std::string get_operands_kind_type_str();
            MethodID *get_operands_kind_method();
            std::string get_operands_kind_method_str();
            std::uint8_t get_operand_register(std::uint8_t index);

        private:
            std::uint8_t array_size;
            std::uint16_t type_index;
            std::vector<std::uint8_t> registers;
        };

        class Instruction3rc : public Instruction
        /***
         * Construct array of given type and size,
         * filling it with supplied contents.
         * 
         * Example instruction:
         *   	op {vCCCC .. vNNNN}, meth@BBBB
         *      op {vCCCC .. vNNNN}, site@BBBB
         *      op {vCCCC .. vNNNN}, type@BBBB
         * 
         * A: array size and argument word count (8 bits).
         * B: type index (16 bits).
         * C: first argument register (16 bits).
         * N = A + C - 1
         */
        {
        public:
            Instruction3rc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction3rc();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::uint8_t get_array_size();

            std::uint16_t get_index();

            DVMTypes::Operand get_operands_types();

            DVMTypes::Kind get_index_kind();

            std::any get_last_operand();
            std::string get_last_operand_str();

            Type *get_operands_type();
            std::string get_operands_type_str();

            MethodID *get_operands_method();
            std::string get_operands_method_str();

            std::uint8_t get_operand_register(std::uint8_t index);

        private:
            std::uint8_t array_size;
            std::uint16_t index;
            std::vector<std::uint8_t> registers;
        };

        class Instruction45cc : public Instruction
        /***
         * Invoke the indicated signature polymorphic method.
         * The result (if any) may be stored with an appropriate 
         * move-result* variant as the immediately subsequent instruction.
         * 
         * Example Instruction:
         *  invoke-polymorphic {vC, vD, vE, vF, vG}, meth@BBBB, proto@HHHH
         * 
         * A: argument word count (4 bits).
         * BBBB: method reference index (16 bits)
         * C: receiver (4 bits).
         * D..G: argument registers (4 bits each).
         * H: prototype reference index (16 bits).
         */
        {
        public:
            Instruction45cc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction45cc();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::uint8_t get_reg_count();
            DVMTypes::Operand get_register_types();
            std::uint8_t get_register(std::uint16_t index);

            DVMTypes::Kind get_method_ref_kind();
            MethodID *get_method_ref();
            std::string get_method_ref_str();

            DVMTypes::Kind get_proto_ref_kind();
            ProtoID *get_proto_ref();
            std::string get_proto_ref_str();

        private:
            std::uint8_t reg_count;
            std::vector<std::uint8_t> registers;
            std::uint16_t method_reference;
            std::uint16_t proto_reference;
        };

        class Instruction4rcc : public Instruction
        /***
         * Invoke the indicated method handle.
         * 
         * Example Instruction:
         *  invoke-polymorphic/range {vCCCC .. vNNNN}, meth@BBBB, proto@HHHH
         * 
         * A: argument workd count (8 bits)
         * B: method reference index (16 bits)
         * C: receiver (16 bits).
         * H: prototype reference index (16 bits).
         * N = A + C - 1
         */
        {
        public:
            Instruction4rcc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction4rcc();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::uint8_t get_reg_count();
            DVMTypes::Operand get_register_types();
            std::uint16_t get_register(std::uint16_t index);

            DVMTypes::Kind get_method_ref_kind();
            MethodID *get_method_ref();
            std::string get_method_ref_str();

            DVMTypes::Kind get_proto_ref_kind();
            ProtoID *get_proto_ref();
            std::string get_proto_ref_str();

        private:
            std::uint8_t reg_count;
            std::vector<std::uint16_t> registers;
            std::uint16_t method_reference;
            std::uint16_t proto_reference;
        };

        class Instruction51l : public Instruction
        /***
         * Move given literal value into specified register pair
         * 
         * Example Instruction:
         * 
         * const-wide vAA, #+BBBBBBBBBBBBBBBB
         * 
         * vAA: destination register (8 bits)
         * #+BBBBBBBBBBBBBBBB: arbitrary double-width constant (64 bits)
         */
        {
        public:
            Instruction51l(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~Instruction51l();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            DVMTypes::Operand get_source_type();
            std::uint64_t get_source();
            DVMTypes::Operand get_destination_type();
            std::uint8_t get_destination();

        private:
            std::uint8_t vAA;
            std::uint64_t nBBBBBBBBBBBBBBBB;
        };

        class PackedSwitch : public Instruction
        /***
         * Packed switch, present in methods which
         * make use of this kind of data.
         */
        {
        public:
            PackedSwitch(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~PackedSwitch();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::int32_t get_first_key();
            std::vector<std::int32_t> get_targets();

        private:
            std::uint16_t ident;
            std::uint16_t size;
            std::int32_t first_key;
            std::vector<std::int32_t> targets;
        };

        class SparseSwitch : public Instruction
        /***
         * Sparse switch, present in methods which
         * make use of this kind of data.
         */
        {
        public:
            SparseSwitch(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~SparseSwitch();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::int32_t get_target_by_key(std::int32_t key);
            std::int32_t get_key_by_pos(size_t pos);
            std::int32_t get_target_by_pos(size_t pos);

            std::vector<std::tuple<std::int32_t, std::int32_t>> get_keys_targets();

        private:
            std::uint16_t ident;
            std::uint16_t size;
            std::vector<std::tuple<std::int32_t, std::int32_t>> keys_targets;
        };

        class FillArrayData : public Instruction
        /***
         * Class present in methods which make use of
         * this kind of data.
         */
        {
        public:
            FillArrayData(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
            ~FillArrayData();

            virtual std::string get_output();
            virtual std::uint64_t get_raw();
            virtual std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> get_operands();

            std::vector<std::uint8_t> get_data();

        private:
            std::uint16_t ident;
            std::uint16_t element_width;
            std::uint32_t size;
            std::vector<std::uint8_t> data;
        };

        typedef struct handler_data_
        {
            std::string handler_type;
            std::uint64_t handler_start_addr;
            std::vector<std::any> basic_blocks;
        } handler_data;

        typedef struct exceptions_data_
        {
            std::uint64_t try_value_start_addr;
            std::uint64_t try_value_end_addr;
            std::vector<handler_data> handler;
        } exceptions_data;

        std::shared_ptr<Instruction> get_instruction_object(std::uint32_t opcode, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file);
        std::vector<std::int64_t> determine_next(std::shared_ptr<Instruction> instr,
                                                                  std::uint64_t curr_idx,
                                                                  std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions);
        std::vector<exceptions_data> determine_exception(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::shared_ptr<EncodedMethod> method);
    }
}

#endif