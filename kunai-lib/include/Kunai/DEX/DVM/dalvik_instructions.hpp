//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dalvik_instructions.hpp
// @brief Definition of all the instructions available on the Dalvik
// Virtual Machine
#ifndef KUNAI_DEX_DVM_DALVIK_INSTRUCTIONS_HPP
#define KUNAI_DEX_DVM_DALVIK_INSTRUCTIONS_HPP

#include "Kunai/Utils/kunaistream.hpp"
#include "Kunai/DEX/parser/parser.hpp"
#include "Kunai/DEX/DVM/dvm_types.hpp"
#include "Kunai/DEX/DVM/dalvik_opcodes.hpp"

#include <iostream>
#include <span>
#include <string>
#include <variant>

namespace KUNAI
{
namespace DEX
{
    /// @brief Type of dex instruction, this can be used to check
    /// what kind of instruction is the current one, in order to avoid
    /// using dynamic casting
    enum class dexinsttype_t
    {
        DEX_INSTRUCTION00X,
        DEX_INSTRUCTION10X,
        DEX_INSTRUCTION12X,
        DEX_INSTRUCTION11N,
        DEX_INSTRUCTION11X,
        DEX_INSTRUCTION10T,
        DEX_INSTRUCTION20T,
        DEX_INSTRUCTION20BC,
        DEX_INSTRUCTION22X,
        DEX_INSTRUCTION21T,
        DEX_INSTRUCTION21S,
        DEX_INSTRUCTION21H,
        DEX_INSTRUCTION21C,
        DEX_INSTRUCTION23X,
        DEX_INSTRUCTION22B,
        DEX_INSTRUCTION22T,
        DEX_INSTRUCTION22S,
        DEX_INSTRUCTION22C,
        DEX_INSTRUCTION22CS,
        DEX_INSTRUCTION30T,
        DEX_INSTRUCTION32X,
        DEX_INSTRUCTION31I,
        DEX_INSTRUCTION31T,
        DEX_INSTRUCTION31C,
        DEX_INSTRUCTION35C,
        DEX_INSTRUCTION3RC,
        DEX_INSTRUCTION45CC,
        DEX_INSTRUCTION4RCC,
        DEX_INSTRUCTION51L,
        DEX_PACKEDSWITCH,
        DEX_SPARSESWITCH,
        DEX_FILLARRAYDATA,
        DEX_DALVIKINCORRECT,
        DEX_NONE_OP = 99,
    };

    /// @brief Base class for the Instructions of the Dalvik Bytecode
    class Instruction
    {
        /// @brief Instruction type from the enum
        dexinsttype_t instruction_type;

    protected:
        /// @brief Opcodes of the instruction
        std::span<std::uint8_t> op_codes;
        /// @brief Length of the instruction
        std::uint32_t length;
        /// @brief op code from the instruction
        std::uint32_t op;
        /// @brief address of the instruction
        std::uint64_t address;

    public:
        /// @brief Constructor of the Instruction, here is applied
        /// the parsing of the opcodes
        /// @param bytecode
        /// @param index
        /// @param instruction_type
        Instruction(std::vector<uint8_t> &bytecode, std::size_t index, dexinsttype_t instruction_type)
            : instruction_type(instruction_type), length(0), op(0), op_codes({})
        {
        }

        Instruction(std::vector<uint8_t> &bytecode, std::size_t index, dexinsttype_t instruction_type, std::uint32_t length)
            : instruction_type(instruction_type), length(length), op(0)
        {
            /// op_codes we can read here for all the classes
            /// that derives from Instruction, we have that is
            /// the bytecode from the index to index+length
            op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};
        }

        /// @brief Destructor of the instruction
        virtual ~Instruction() = default;

        /// @brief Get the kind of instruction, use a DalvikOpcodes function
        /// @return TYPES::Kind of the instruction
        virtual TYPES::Kind get_kind() const
        {
            return DalvikOpcodes::get_instruction_type(op);
        }
        /// @brief Get the instruction type from the enum
        /// @return dex instruction type
        virtual dexinsttype_t get_instruction_type() const
        {
            return instruction_type;
        }

        /// @brief Get the length of the instruction
        /// @return current length of the instruction
        virtual std::uint32_t get_instruction_length() const
        {
            return length;
        }

        /// @brief Get the opcode of the instruction
        /// @return opcode of the instruction
        virtual std::uint32_t get_instruction_opcode() const
        {
            return op;
        }

        /// @brief Set the address of the instruction
        /// @param address new address of the instruction
        virtual void set_address(std::uint64_t address)
        {
            this->address = address;
        }

        /// @brief Get the address of the instruction
        /// @return address of the instruction
        virtual std::uint64_t get_address() const
        {
            return address;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return "";
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << "";
        }

        /// @brief Return the op codes in raw from the instruction
        /// @return constant reference with op codes in raw
        virtual const std::span<std::uint8_t> &get_opcodes()
        {
            return op_codes;
        }

        /// @brief Check if the instruction is a terminator (branch, ret, multibranch)
        /// @return true if instruction is a terminator instruction
        virtual bool is_terminator()
        {
            auto operation = DalvikOpcodes::get_instruction_operation(op);

            if (operation == TYPES::Operation::CONDITIONAL_BRANCH_DVM_OPCODE ||
                operation == TYPES::Operation::UNCONDITIONAL_BRANCH_DVM_OPCODE ||
                operation == TYPES::Operation::RET_BRANCH_DVM_OPCODE ||
                operation == TYPES::Operation::MULTI_BRANCH_DVM_OPCODE)
                return true;
            return false;
        }
    };

    /// @brief Useless instruction with opcode of 00
    /// no instruction represents this, it's not either a nop
    class Instruction00x : public Instruction
    {
    public:
        /// @brief Constructor of Instruction00x this instruction does nothing
        /// @param bytecode bytecode with the opcodes
        /// @param index
        Instruction00x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser) : Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION00X)
        {
        }
    };

    /// @brief Instruction for wasting cycles. It represents
    /// a nop, it has a length of 2
    class Instruction10x : public Instruction
    {
    public:
        Instruction10x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op);
        }
    };

    /// @brief Move the contents of one register to another
    /// length of the instruction is 2 bytes, it contains
    /// two registers vA and vB of 4 bits each one
    class Instruction12x : public Instruction
    {
        /// @brief destination register
        std::uint8_t vA;
        /// @brief source register
        std::uint8_t vB;

    public:
        Instruction12x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the index of the destination register
        /// @return index of destination register
        std::uint8_t get_destination() const
        {
            return vA;
        }

        /// @brief Get the index of the source register
        /// @return index of source register
        std::uint8_t get_source() const
        {
            return vB;
        }

        /// @brief Get the operand type of the destination
        /// @return operand type of destination
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the operand type of the source
        /// @return operand type of source
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    "v" + std::to_string(vB) + ", " +
                    "v" + std::to_string(vA);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        "v" + std::to_string(vB) + ", " +
                        "v" + std::to_string(vA);
        }
    };

    /// @brief Instruction for moving a given literal, the
    /// instruction has a register and a literal value, with
    /// a size of 2 bytes of instruction
    class Instruction11n : public Instruction
    {
        /// @brief destination register
        std::uint8_t vA;
        /// @brief Literal value of instruction
        std::int8_t nB;

    public:
        Instruction11n(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint8_t get_destination() const
        {
            return vA;
        }

        std::int8_t get_source() const
        {
            return nB;
        }

        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    "v" + std::to_string(vA) + ", " +
                    std::to_string(nB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        "v" + std::to_string(vA) + ", " +
                        std::to_string(nB);
        }
    };

    /// @brief Move single, double-words or objects from
    /// invoke results, also save caught exception into
    /// given register
    class Instruction11x : public Instruction
    {
        /// @brief  destination of move
        std::uint8_t vAA;

    public:
        Instruction11x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get destination register index of the operation
        /// @return index of register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the type of operand for the destination
        /// @return operand type of register
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    "v" + std::to_string(vAA);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        "v" + std::to_string(vAA);
        }
    };

    /// @brief Unconditional jump instruction. An offset
    /// is given to know where to jump
    class Instruction10t : public Instruction
    {
        /// @brief Offset where to jump with unconditional jump
        std::int8_t nAA;

    public:
        Instruction10t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get offset of the jump
        /// @return offset of jump instruction
        std::int8_t get_jump_offset() const
        {
            return nAA;
        }

        /// @brief Get type of operand in this case an offset
        /// @return return offset type
        TYPES::Operand get_operand_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    std::to_string(nAA);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        std::to_string(nAA);
        }
    };

    /// @brief Another unconditional jump with a bigger offset
    /// of 2 bytes for the offset
    class Instruction20t : public Instruction
    {
        /// @brief Offset where to jump
        std::int16_t nAAAA;

    public:
        Instruction20t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the offset where to jump with an unconditional jump
        /// @return offset of the jump
        std::int16_t get_offset() const
        {
            return nAAAA;
        }

        /// @brief Get the type of the operand of this instruction (jump)
        /// @return offset type
        TYPES::Operand get_operand_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    std::to_string(nAAAA);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        std::to_string(nAAAA);
        }
    };

    /// @brief opAA, kind@BBBB, where AA indicates a type of error
    /// and BBBB and index into the appropiate table
    class Instruction20bc : public Instruction
    {
        /// @brief type of error
        std::uint8_t nAA;
        /// @brief index into appropiate table
        std::uint16_t nBBBB;

    public:
        Instruction20bc(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the index of the type of error
        /// @return index of error
        std::uint8_t get_type_of_error_data() const
        {
            return nAA;
        }

        /// @brief Get the type of operand for the error operand
        /// @return literal type
        TYPES::Operand get_error_operand_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Get the index to the appropiate table of the instruction...
        /// @return I don't really understand this instruction...
        std::uint16_t get_index_into_table() const
        {
            return nBBBB;
        }

        TYPES::Operand get_index_operand_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    std::to_string(nAA) + ", kind@" + std::to_string(nBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        std::to_string(nAA) + ", kind@" + std::to_string(nBBBB);
        }
    };

    /// @brief Move the contents of one non-object register to another.
    /// an instruction like move/from16 vAA, vBBBB where vAA is 8 bits,
    /// and vBBBB is 16 bits
    class Instruction22x : public Instruction
    {
        /// @brief destination register (8 bits)
        std::uint8_t vAA;
        /// @brief source register (16 bits)
        std::uint16_t vBBBB;

    public:
        Instruction22x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get index of the register of destination
        /// @return index of destination register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the type of operand from the destination
        /// @return operand type of destination
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the index of the register of the source
        /// @return index of source register
        std::uint16_t get_source() const
        {
            return vBBBB;
        }

        /// @brief Get the type of operand from the source
        /// @return operand type of source
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " " +
                    "v" + std::to_string(vAA) + ", " +
                    std::to_string(vBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                        "v" + std::to_string(vAA) + ", " +
                        std::to_string(vBBBB);
        }
    };

    /// @brief Branch to the given destination if the given
    /// register's value compares with 0 as specified.
    /// Example: if-testz vAA, +BBBB where vAA is the register
    /// to test (8 bits) and +BBBB the offset (16 bits)
    class Instruction21t : public Instruction
    {
        /// @brief Register to check against zero
        std::uint8_t vAA;
        /// @brief Offset where to jump if-zero
        std::int16_t nBBBB;

    public:
        Instruction21t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the register used for the check in the jump
        /// @return register checked
        std::uint8_t get_check_reg() const
        {
            return vAA;
        }

        /// @brief Get the type of the checked register
        /// @return type register
        TYPES::Operand get_check_reg_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the offset of the jump
        /// @return offset of jump
        std::int16_t get_jump_offset() const
        {
            return nBBBB;
        }

        /// @brief Get the type of the offset of the jump
        /// @return type of offset
        TYPES::Operand get_offset_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vAA) + ", " +
                    std::to_string(nBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vAA) + ", " +
                        std::to_string(nBBBB);
        }
    };

    /// @brief Move given literal value into specified register.
    /// Example of instruction: const/16 vAA, #+BBBB. Where
    /// vAA is the destination register and #+BBBB is the literal
    /// moved
    class Instruction21s : public Instruction
    {
        /// @brief destination register
        std::uint8_t vAA;
        /// @brief literal value
        std::int16_t nBBBB;

    public:
        Instruction21s(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the index of the destination register
        /// @return index of destination register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the destination type of the instruction
        /// @return destination type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the source value of the instruction
        /// @return source value
        std::int16_t get_source() const
        {
            return nBBBB;
        }

        /// @brief Get the source type of the instruction
        /// @return source type
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vAA) + ", " +
                    std::to_string(nBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vAA) + ", " +
                        std::to_string(nBBBB);
        }
    };

    /// @brief Move given literal value into specified register.
    /// Example: const/high16 vAA, #+BBBB0000 where vAA is the
    /// destination register (8 bits) and  #+BBBB0000: signed int (16 bits)
    class Instruction21h : public Instruction
    {
        /// @brief Destination register
        std::uint8_t vAA;
        /// @brief source value
        std::int64_t nBBBB;

    public:
        Instruction21h(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the index of the destination register
        /// @return index of destination register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the destination type of the instruction
        /// @return destination type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the source value of the instruction
        /// @return source value
        std::int64_t get_source() const
        {
            return nBBBB;
        }

        /// @brief Get the source type of the instruction
        /// @return source type
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vAA) + ", " +
                    std::to_string(nBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vAA) + ", " +
                        std::to_string(nBBBB);
        }
    };

    /// @brief Move a reference to a register from a string, type, etc
    /// example instruction: const-string vAA, string@BBBB
    class Instruction21c : public Instruction
    {
        /// @brief Is the value a string?
        bool is_str = false;
        /// @brief Is the value a type?
        bool is_type = false;
        /// @brief In case of type, what kind of type
        bool is_fundamental = false;
        bool is_array = false;
        bool is_class = false;
        bool is_unknown = false;

        /// @brief is a field?
        bool is_field = false;
        /// @brief is a method?
        bool is_method = false;
        /// @brief is a proto?
        bool is_proto = false;

        /// @brief destination register (8 bits)
        std::uint8_t vAA;
        /// @brief source id, this can be a string, type, etc (16 bits)
        std::uint16_t iBBBB;
        /// @brief Source representation of the source
        std::string source_str;

        /// @brief pointer to DEX parser to access some of the information
        Parser *parser;

    public:
        Instruction21c(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser);

        /// @brief Get the index of the register for destination
        /// @return index of register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the type of the destination
        /// @return return register operand type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the index used as source operand,
        /// this is an index to a string, type, etc...
        /// @return value of index
        std::uint16_t get_source() const
        {
            return iBBBB;
        }
        /// @brief Get the type of the source, this time is a KIND
        /// the KIND can be various things
        /// @return return KIND type
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::KIND;
        }

        /// @brief Print a string version of the source
        /// @return constant reference to string of source
        const std::string &pretty_print_source() const
        {
            return source_str;
        }

        /// @brief Check if source is a string
        /// @return true in case source is a string
        bool is_source_string() const
        {
            return is_str;
        }

        /// @brief get reference of the string this should be
        /// called only if is_str == true
        /// @return string reference
        std::string &get_source_str()
        {
            return source_str;
        }

        /// @brief Check if source is a DVMType and return a pointer
        /// @return pointer to DVMType or nullptr
        DVMType *get_source_dvmtype() const
        {
            if (is_type)
                return parser->get_types().get_type_from_order(iBBBB);
            return nullptr;
        }

        /// @brief Check if source is a DVMFundamental and return a pointer
        /// @return pointer to DVMFundamental or nullptr
        DVMFundamental *get_source_dvmfundamental() const
        {
            if (is_fundamental)
                return reinterpret_cast<DVMFundamental *>(
                    parser->get_types().get_type_from_order(iBBBB));
            return nullptr;
        }

        /// @brief check if source is a DVMClass and return a pointer
        /// @return pointer to DVMClass or nullptr
        DVMClass *get_source_dvmclass() const
        {
            if (is_class)
                return reinterpret_cast<DVMClass *>(
                    parser->get_types().get_type_from_order(iBBBB));
            return nullptr;
        }

        /// @brief check if source is a DVMArray and return a pointer
        /// @return pointer to DVMArray or nullptr
        DVMArray *get_source_dvmarray() const
        {
            if (is_array)
                return reinterpret_cast<DVMArray *>(
                    parser->get_types().get_type_from_order(iBBBB));
            return nullptr;
        }

        /// @brief check if source is a FieldID and return a pointer
        /// @return pointer to FieldID or nullptr
        FieldID *get_source_field() const
        {
            if (is_field)
                return parser->get_fields().get_field(iBBBB);
            return nullptr;
        }

        /// @brief Check if source is a ProtoID and return a pointer
        /// @return pointer to ProtoID or nullptr
        ProtoID *get_source_proto() const
        {
            if (is_proto)
                return parser->get_protos().get_proto_by_order(iBBBB);
            return nullptr;
        }

        /// @brief check if source is a MethodID and return a pointer
        /// @return pointer to MethodID or nullptr
        MethodID *get_source_method() const
        {
            if (is_method)
                return parser->get_methods().get_method(iBBBB);
            return nullptr;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vAA) + ", " +
                    source_str + "(" + std::to_string(iBBBB) + ")";
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vAA) + ", " +
                        source_str + "(" + std::to_string(iBBBB) + ")";
        }
    };

    /// @brief Perform indicated floating point or long comparison
    /// Example: cmpkind vAA, vBB, vCC
    class Instruction23x : public Instruction
    {
        /// @brief destination register (8 bits).
        std::uint8_t vAA;
        /// @brief first source register or pair (8 bits).
        std::uint8_t vBB;
        /// @brief second source register or pair (8 bits).
        std::uint8_t vCC;

    public:
        Instruction23x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the register for the destination
        /// @return destination register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the type of the destination
        /// @return get REGISTER operand
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the register for the first source
        /// @return first source register
        std::uint8_t get_first_source() const
        {
            return vBB;
        }

        /// @brief Get the type of the first source
        /// @return get REGISTER operand
        TYPES::Operand get_first_source_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the register for the second source
        /// @return second source register
        std::uint8_t get_second_source() const
        {
            return vCC;
        }

        /// @brief Get the type of the second source
        /// @return get REGISTER operand
        TYPES::Operand get_second_source_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vAA) + ", v" +
                    std::to_string(vBB) + ", v" +
                    std::to_string(vCC);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vAA) + ", v" +
                        std::to_string(vBB) + ", v" +
                        std::to_string(vCC);
        }
    };

    /// @brief Perform indicated binary operation on the indicated
    /// register and literal value, storing result in destination
    /// register. Example: add-int/lit8 vAA, vBB, #+CC
    /// Semantic of the instruction: vAA = vBB + #+CC
    class Instruction22b : public Instruction
    {
        /// @brief Destination register (8 bits)
        std::uint8_t vAA;
        /// @brief First operand (8 bits)
        std::uint8_t vBB;
        /// @brief Second operand (8 bits)
        std::int8_t nCC;

    public:
        Instruction22b(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the index value of the destination register
        /// @return register index
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the type of the destination
        /// @return return REGISTER type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the first operand of the instruction
        /// @return index of register operand
        std::uint8_t get_first_operand() const
        {
            return vBB;
        }

        /// @brief Get the type of the first operand
        /// @return return REGISTER type
        TYPES::Operand get_first_operand_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the value of the second operand
        /// @return value of second operand
        std::int8_t get_second_operand() const
        {
            return nCC;
        }

        /// @brief Get the type of the second operand
        /// @return return LITERAL type
        TYPES::Operand get_second_operand_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vAA) + ", v" +
                    std::to_string(vBB) + ", " +
                    std::to_string(nCC);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vAA) + ", v" +
                        std::to_string(vBB) + ", " +
                        std::to_string(nCC);
        }
    };

    /// @brief Branch to given offset after comparison of two registers.
    /// Example if-test vA, vB, +CCCC
    class Instruction22t : public Instruction
    {
        /// @brief First register checked (4 bits)
        std::uint8_t vA;
        /// @brief Second register checked (4 bits)
        std::uint8_t vB;
        /// @brief Offset where to jump
        std::int16_t nCCCC;

    public:
        Instruction22t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the first operand of the check
        /// @return index of register
        std::uint8_t get_first_operand() const
        {
            return vA;
        }

        /// @brief Get the type of the first operand of the comparison
        /// @return return REGISTER type
        TYPES::Operand get_first_operand_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the second operand of the check
        /// @return index of register
        std::uint8_t get_second_operand() const
        {
            return vB;
        }

        /// @brief Get the type of the second operand of the comparison
        /// @return return REGISTER type
        TYPES::Operand get_second_operand_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the offset of the jump in case this is taken
        /// @return offset for the conditional jump
        std::int16_t get_offset() const
        {
            return nCCCC;
        }

        /// @brief Get the type of the offset for the jump
        /// @return return OFFSET type
        TYPES::Operand get_offset_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vA) + ", v" +
                    std::to_string(vB) + ", " +
                    std::to_string(nCCCC);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vA) + ", v" +
                        std::to_string(vB) + ", " +
                        std::to_string(nCCCC);
        }
    };

    /// @brief Perform indicated binary operation on the operands
    /// storing finally the result in the destination register.
    /// Example: add-int/lit16 vA, vB, #+CCCC
    /// Semantic: vA = vB + #+CCCC
    class Instruction22s : public Instruction
    {
        /// @brief destination regsiter (4 bits)
        std::uint8_t vA;
        /// @brief first operand (4 bits)
        std::uint8_t vB;
        /// @brief second operand (16 bits)
        std::int16_t nCCCC;

    public:
        Instruction22s(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the destination of the operation
        /// @return index of the destination register
        std::uint8_t get_destination() const
        {
            return vA;
        }

        /// @brief Get the type of the operand used for destination
        /// @return get REGISTER type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the first operand of the instruction
        /// @return index of register for operand
        std::uint8_t get_first_operand() const
        {
            return vB;
        }

        /// @brief Get the type of the first operand of the instruction
        /// @return return REGISTER type
        TYPES::Operand get_first_operand_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the second operand of the instruction
        /// @return literal value used in the instruction
        std::int16_t get_second_operand() const
        {
            return nCCCC;
        }

        /// @brief Get the type of the second operand of the instruction
        /// @return return LITERAL type
        TYPES::Operand get_second_operand_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vA) + ", v" +
                    std::to_string(vB) + ", " +
                    std::to_string(nCCCC);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vA) + ", v" +
                        std::to_string(vB) + ", " +
                        std::to_string(nCCCC);
        }
    };

    /// @brief Store in the given destination 1 if the register
    /// provided contains an instance of the given type/field,
    /// 0 in other case.
    /// Example: instance-of vA, vB, type@CCCC
    /// Semantic: vA = type(vB) == type@CCCC ? 1 : 0
    class Instruction22c : public Instruction
    {
        /// @brief Destination register (4 bits)
        std::uint8_t vA;
        /// @brief Register with type to check (4 bits)
        std::uint8_t vB;
        /// @brief Type/FieldID to check
        std::uint16_t iCCCC;
        /// @brief string representation of the type or field
        std::string iCCCC_str;
        /// @brief parser for obtaining the values
        Parser *parser;
        /// @brief Check if current value is a type
        bool is_type = false;
        /// @brief Check if current value is a field
        bool is_field = false;

    public:
        Instruction22c(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser);

        /// @brief Get the destination operand for the instruction
        /// @return index of the register for the destination
        std::uint8_t get_destination() const
        {
            return vA;
        }

        /// @brief Get the destination operand type
        /// @return return REGISTER type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the operand checked in the instruction
        /// @return index of the register checked
        std::uint8_t get_operand() const
        {
            return vB;
        }

        /// @brief Get the type of the operand of the instruction
        /// @return return REGISTER type
        TYPES::Operand get_operand_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the ID of the checked Type/Field
        /// @return ID of checked Type/Field
        std::uint16_t get_checked_id() const
        {
            return iCCCC;
        }

        /// @brief Get the type of the checked ID
        /// @return return KIND type
        TYPES::Operand get_checked_id_type() const
        {
            return TYPES::Operand::KIND;
        }

        /// @brief Get a pretty-printed version of the checked value
        /// @return string version of checked type/field
        const std::string &get_checked_value_str() const
        {
            return iCCCC_str;
        }

        /// @brief Check if checked value is a DVMType and get a pointer
        /// @return pointer to DVMType or nullptr
        DVMType *get_checked_dvmtype() const
        {
            if (is_type)
                return parser->get_types().get_type_from_order(iCCCC);
            return nullptr;
        }

        /// @brief Check if checked value is a FieldID and get a pointer
        /// @return pointer to FieldID or nullptr
        FieldID *get_checked_field()
        {
            if (is_field)
                return parser->get_fields().get_field(iCCCC);
            return nullptr;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vA) + ", v" + std::to_string(vB) + ", " +
                    iCCCC_str + "(" + std::to_string(iCCCC) + ")";
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vA) + ", v" + std::to_string(vB) + ", " +
                        iCCCC_str + "(" + std::to_string(iCCCC) + ")";
        }
    };

    /// @brief Format suggested for statically linked field access
    /// instructions or Types. Example: op vA, vB, fieldoff@CCCC
    /// *-QUICK methods
    class Instruction22cs : public Instruction
    {
        /// @brief Maybe destination?
        std::uint8_t vA;
        /// @brief Maybe where field is?
        std::uint8_t vB;
        /// @brief the field offset
        std::uint16_t iCCCC;
        /// @brief field as string
        std::string iCCCC_str;
        /// @brief is a field?
        bool is_field = false;
        /// @brief parser to obtain information
        Parser *parser;

    public:
        Instruction22cs(std::vector<uint8_t> &bytecode, std::size_t index, Parser *parser);

        /// @brief Get the index of the first register used in the instruction
        /// @return value of register A
        std::uint8_t get_register_A() const
        {
            return vA;
        }

        /// @brief Get the type for the register A
        /// @return return REGISTER type
        TYPES::Operand get_register_A_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the index of the second register used in the instruction
        /// @return value of register B
        std::uint8_t get_register_B() const
        {
            return vB;
        }

        /// @brief Get the type for the register B
        /// @return return REGISTER type
        TYPES::Operand get_register_B_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the offset for the field
        /// @return int value with field for offset
        std::uint16_t get_field_offset() const
        {
            return iCCCC;
        }

        /// @brief Get the type for the offset, probably KIND
        /// @return return KIND type (I think is that one...)
        TYPES::Operand get_field_offset_type() const
        {
            return TYPES::Operand::KIND;
        }

        /// @brief Get a string representation of the Field
        /// @return string representation of field
        const std::string &get_field_string() const
        {
            return iCCCC_str;
        }

        /// @brief Check if the idx is from a field and return a FieldID
        /// @return pointer to FieldID or nullptr
        FieldID *get_field() const
        {
            if (is_field)
                return parser->get_fields().get_field(iCCCC);
            return nullptr;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" +
                    std::to_string(vA) + ", v" + std::to_string(vB) + ", " +
                    iCCCC_str + "(" + std::to_string(iCCCC) + ")";
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" +
                        std::to_string(vA) + ", v" + std::to_string(vB) + ", " +
                        iCCCC_str + "(" + std::to_string(iCCCC) + ")";
        }
    };

    /// @brief Unconditional jump to indicated offset
    /// Example: goto/32 +AAAAAAAA
    class Instruction30t : public Instruction
    {
        /// @brief offset where to jump in the instruction (32 bits)
        std::int32_t nAAAAAAAA;

    public:
        Instruction30t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the offset of the jump
        /// @return offset of unconditional jump
        std::int32_t get_offset() const
        {
            return nAAAAAAAA;
        }

        /// @brief Get the type of the offset
        /// @return return OFFSET of the jump
        TYPES::Operand get_offset_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + std::to_string(nAAAAAAAA);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + std::to_string(nAAAAAAAA);
        }
    };

    /// @brief Binary operation between registers of 16 bits
    /// Example: move/16 vAAAA, vBBBB
    class Instruction32x : public Instruction
    {
        /// @brief Destination register (16 bits)
        std::uint16_t vAAAA;
        /// @brief Source register (16 bits)
        std::uint16_t vBBBB;
    public:
        Instruction32x(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the destination operand of the instruction
        /// @return index of register destination
        std::uint16_t get_destination() const
        {
            return vAAAA;
        }

        /// @brief Get the type of the destination operand
        /// @return return REGISTER type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the source operand of the instruction
        /// @return index of register source
        std::uint16_t get_source() const
        {
            return vBBBB;
        }

        /// @brief Get the type of the source operand
        /// @return return REGISTER type
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAAAA) + ", v" + std::to_string(vBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAAAA) + ", v" + std::to_string(vBBBB);
        }
    };

    /// @brief Instructions between a register and
    /// a literal value of 32 bits.
    /// Example: const vAA, #+BBBBBBBB
    class Instruction31i : public Instruction
    {
        /// @brief destination register (8 bits)
        std::uint8_t vAA;
        /// @brief source value (32 bits)
        std::uint32_t nBBBBBBBB;
    public:
        Instruction31i(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the destination operand of the instruction
        /// @return index of destination register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the destination operand type of the instruction
        /// @return return REGISTER type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the source operand of the instruction
        /// @return value of source operand
        std::uint32_t get_source() const
        {
            return nBBBBBBBB;
        }

        /// @brief Get the source operand type of the instruction
        /// @return return LITERAL type
        TYPES::Operand get_source_type() const
        {
            return TYPES::Operand::LITERAL;
        }
        
        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAA) + ", " + std::to_string(nBBBBBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAA) + ", " + std::to_string(nBBBBBBBB);
        }
    };

    /// Forward declaration of different type of switch
    class PackedSwitch;
    class SparseSwitch;

    /// @brief Fill given array with indicated data. Reference
    /// must be an array of primitives. Also used for specifying
    /// switch tables
    /// Example: fill-array-data vAA, +BBBBBBBB
    class Instruction31t : public Instruction
    {
    public:
        /// @brief Enum specifying the type of switch
        /// for the data table
        enum type_of_switch_t
        {
            PACKED_SWITCH = 0,
            SPARSE_SWITCH,
            NONE_SWITCH
        };
    private:
        /// @brief array reference (8 bits)
        std::uint8_t vAA;
        /// @brief signed "branch" offset to table data pseudo instruction (32 bits)
        std::int32_t nBBBBBBBB;
        
        /// @brief type of switch in case it is a switch
        type_of_switch_t type_of_switch;

        /// @brief pointer to PackedSwitch in case is this
        PackedSwitch * packed_switch = nullptr;
        /// @brief pointer to SparseSwitch in case is this
        SparseSwitch * sparse_switch = nullptr;
    public:
        Instruction31t(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief get the register used as reference for switch/array
        /// @return index of register for reference
        std::uint8_t get_ref_register() const
        {
            return vAA;
        }

        /// @brief Get the type of the reference register
        /// @return return REGISTER type
        TYPES::Operand get_ref_register_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Return the offset to the table with packed data
        /// @return offset to packed data
        std::int32_t get_offset() const
        {
            return nBBBBBBBB;
        }

        /// @brief Return the type of the offset
        /// @return return OFFSET type
        TYPES::Operand get_offset_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Get the type of switch in case the instruction
        /// is a switch
        /// @return type of switch value
        type_of_switch_t get_type_of_switch() const
        {
            return type_of_switch;
        }

        /// @brief Get the pointer to a packed switch in case it exists
        /// @return pointer to PackedSwitch
        PackedSwitch * get_packed_switch()
        {
            return packed_switch;
        }

        /// @brief Get the pointer to sparse switch in case it exists
        /// @return pointer to SparseSwitch
        SparseSwitch * get_sparse_switch()
        {
            return sparse_switch;
        }

        /// @brief Set the pointer to the PackedSwitch
        /// @param packed_switch possible instruction pointed
        void set_packed_switch(PackedSwitch * packed_switch)
        {
            this->packed_switch = packed_switch;
        }

        /// @brief Set the pointer to the SparseSwitch
        /// @param sparse_switch possible instruction pointed
        void set_sparse_switch(SparseSwitch * sparse_switch)
        {
            this->sparse_switch = sparse_switch;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAA) + ", " + std::to_string(nBBBBBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAA) + ", " + std::to_string(nBBBBBBBB);
        }
    };

    /// @brief Move a reference to string specified by given index
    /// into the specified register.
    /// Example: const-string/jumbo vAA, string@BBBBBBBB
    class Instruction31c : public Instruction
    {
        /// @brief Destination register (8 bits)
        std::uint8_t vAA;
        /// @brief String index from source (32 bits)
        std::uint32_t iBBBBBBBB;
        /// @brief string value from the index
        std::string str_value;
    public:
        Instruction31c(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        /// @brief Get the destination register for the string
        /// @return index of destination register
        std::uint8_t get_destination() const
        {
            return vAA;
        }

        /// @brief Get the destination type of the operand
        /// @return return REGISTER type
        TYPES::Operand get_destination_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get the index of the string operand
        /// @return index of string
        std::uint32_t get_string_idx() const
        {
            return iBBBBBBBB;
        }

        /// @brief Get the type from the string operand
        /// @return return OFFSET type
        TYPES::Operand get_string_idx_type() const
        {
            return TYPES::Operand::OFFSET;
        }

        /// @brief Get the value from the string pointed in the instruction
        /// @return constant reference to string value
        const std::string& get_string_value() const
        {
            return str_value;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAA) + ", " + str_value + "(" + std::to_string(iBBBBBBBB) + ")";
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " v" + 
                    std::to_string(vAA) + ", " + str_value + "(" + std::to_string(iBBBBBBBB) + ")";
        }
    };
    
    /// @brief Construct array of given type and size, filling it with supplied
    /// contents. Type must be an array type. Array's contents must be
    /// single-word.
    /// Example: filled-new-array {vC, vD, vE, vF, vG}, type@BBBB
    class Instruction35c : public Instruction
    {
        /// @brief Size of the array of registers (4 bits)
        std::uint8_t array_size;
        /// @brief Type index (16 bits)
        std::uint16_t type_index;
        /// @brief is a type value?
        bool is_type = false;
        /// @brief is a method value?
        bool is_method = false;
        /// @brief value in string format
        std::string type_str;
        /// @brief vector with registers (4 bits each)
        std::vector<std::uint8_t> registers;
        /// @brief Parser for the types
        Parser * parser;
    public:
        Instruction35c(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);
        
        /// @brief Get the number of registers from the instruction
        /// @return array_size value
        std::uint8_t get_number_of_registers() const
        {
            return array_size;
        }

        /// @brief Get a constant reference to the vector with the registers
        /// @return constant reference to registers
        const std::vector<std::uint8_t>& get_registers() const
        {
            return registers;
        }

        /// @brief Get the type of the registers operand
        /// @return return REGISTER type
        TYPES::Operand get_registers_type()
        {
            return TYPES::Operand::REGISTER;
        }

        /// @brief Get a reference to the vector with the registers
        /// @return reference to registers
        std::vector<std::uint8_t>& get_registers()
        {
            return registers;
        }

        /// @brief Get the idx of the type
        /// @return value with the type index
        std::uint16_t get_type_idx() const
        {
            return type_index;
        }

        /// @brief Get the type of the array
        /// @return return KIND type
        TYPES::Operand get_array_type() const
        {
            return TYPES::Operand::KIND;
        }

        /// @brief Get the DVMType of the array type
        /// @return DVMType of array
        DVMType * get_dvmtype()
        {
            if (is_type)
                return parser->get_types().get_type_from_order(type_index);
            return nullptr;
        }

        MethodID * get_method()
        {
            if (is_method)
                return parser->get_methods().get_method(type_index);
            return nullptr;
        }
        
        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::string instruction = DalvikOpcodes::get_instruction_name(op) + " {";

            for(const auto reg : registers)
            {
                instruction += "v" + std::to_string(reg) + ", ";
            }

            if (registers.size() > 0)
                instruction = instruction.substr(0, instruction.size()-2);
            
            instruction += "}, " + type_str;
 
            return instruction;
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << print_instruction();
        }
    };

    /// @brief Construct array of given type and size,
    /// filling it with supplied contents.
    /// Example instructions:
    ///     op {vCCCC .. vNNNN}, meth@BBBB
    ///     op {vCCCC .. vNNNN}, site@BBBB
    ///     op {vCCCC .. vNNNN}, type@BBBB
    class Instruction3rc : public Instruction
    {
        /// @brief  size of the array
        std::uint8_t array_size;
        /// @brief index of meth, type and call site
        std::uint16_t index;
        /// @brief is a method?
        bool is_method = false;
        /// @brief is a type?
        bool is_type = false;
        /// @brief string value of the type
        std::string index_str;
        /// @brief registers, the registers start by
        /// one first argument register of 16 bits
        std::vector<std::uint16_t> registers;
        /// @brief Parser
        Parser * parser;
    public:
        Instruction3rc(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint8_t get_registers_size() const
        {
            return array_size;
        }

        std::uint16_t get_index_value() const
        {
            return index;
        }

        TYPES::Operand get_index_type() const
        {
            return TYPES::Operand::KIND;
        }

        const std::vector<std::uint16_t>& get_registers() const
        {
            return registers;
        }

        std::vector<std::uint16_t>& get_registers()
        {
            return registers;
        }

        DVMType * get_operand_dvmtype()
        {
            if (is_type)
                return parser->get_types().get_type_from_order(index);
            return nullptr;
        }

        MethodID * get_operand_method()
        {
            if (is_method)
                return parser->get_methods().get_method(index);
            return nullptr;
        }

        std::variant<
            DVMType*,
            MethodID*,
            std::uint16_t> get_operand()
        {
            if (is_type)
                return parser->get_types().get_type_from_order(index);
            else if (is_method)
                return parser->get_methods().get_method(index);
            else
                return index;
        }

        const std::string& get_operand_str() const
        {
            return index_str;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::string instruction = DalvikOpcodes::get_instruction_name(op) + " {";

            for(const auto reg : registers)
            {
                instruction += "v" + std::to_string(reg) + ", ";
            }

            if (registers.size() > 0)
                instruction = instruction.substr(0, instruction.size()-2);
            
            instruction += "}, " + index_str;
 
            return instruction;
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << print_instruction();
        }

    };

    /// @brief Invoke indicated signature polymorphic method.
    /// The result (if any) may be stored with an appropiate
    /// move-result* variant as the immediately subsequent
    /// instruction. Example:
    /// invoke-polymorphic {vC, vD, vE, vF, vG}, meth@BBBB, proto@HHHH
    class Instruction45cc : public Instruction
    {
        /// @brief number of registers in the operation
        std::uint8_t reg_count;
        /// @brief registers for the instruction
        std::vector<std::uint8_t> registers;
        /// @brief index to the method called
        std::uint16_t method_reference;
        /// @brief possible method
        MethodID * method_id;
        /// @brief index to the prototype
        std::uint16_t prototype_reference;
        /// @brief possible prototype
        ProtoID * proto_id;
    public:
        Instruction45cc(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint8_t get_number_of_registers() const
        {
            return reg_count;
        }

        const std::vector<std::uint8_t>& get_registers() const
        {
            return registers;
        }

        std::vector<std::uint8_t>& get_registers()
        {
            return registers;
        }

        std::uint16_t get_method_reference() const
        {
            return method_reference;
        }

        std::uint16_t get_prototype_reference() const
        {
            return prototype_reference;
        }

        MethodID * get_method()
        {
            return method_id;
        }

        ProtoID * get_prototype()
        {
            return proto_id;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::string instruction = DalvikOpcodes::get_instruction_name(op) + " {";

            for(const auto reg : registers)
            {
                instruction += "v" + std::to_string(reg) + ", ";
            }

            if (registers.size() > 0)
                instruction = instruction.substr(0, instruction.size()-2);
            
            instruction += "}, ";

            if (method_id)
                instruction += method_id->pretty_method() + ", ";
            if (proto_id)
                instruction += proto_id->get_shorty_idx();
 
            return instruction;
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << print_instruction();
        }
    };

    /// @brief Invoke the method handle indicated,
    /// this time it can provide with a range of arguments
    /// given by a size and an initial register.
    /// Example:
    ///     invoke-polymorphic/range {vCCCC .. vNNNN}, meth@BBBB, proto@HHHH
    class Instruction4rcc : public Instruction
    {
        /// @brief Number of registers
        std::uint8_t reg_count;
        /// @brief Registers of the instruction
        std::vector<std::uint16_t> registers;
        /// @brief method reference
        std::uint16_t method_reference;
        /// @brief MethodID pointer in case exists
        MethodID * method_id;
        /// @brief Prototype reference
        std::uint16_t prototype_reference;
        /// @brief ProtoID pointer in case exists
        ProtoID * prototype_id;
    public:
        Instruction4rcc(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint8_t get_number_of_registers() const
        {
            return reg_count;
        }

        const std::vector<std::uint16_t>& get_registers() const
        {
            return registers;
        }

        std::vector<std::uint16_t>& get_registers()
        {
            return registers;
        }
        
        std::uint16_t get_method_reference() const
        {
            return method_reference;
        }

        MethodID * get_method()
        {
            return method_id;
        }

        std::uint16_t get_prototype_reference() const
        {
            return prototype_reference;
        }

        ProtoID * get_prototype()
        {
            return prototype_id;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::string instruction = DalvikOpcodes::get_instruction_name(op) + " {";

            for(const auto reg : registers)
            {
                instruction += "v" + std::to_string(reg) + ", ";
            }

            if (registers.size() > 0)
                instruction = instruction.substr(0, instruction.size()-2);
            
            instruction += "}, ";

            if (method_id)
                instruction += method_id->pretty_method() + ", ";
            if (prototype_id)
                instruction += prototype_id->get_shorty_idx();
 
            return instruction;
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << print_instruction();
        }
    };
    
    /// @brief Move given literal value into specified register pair
    /// Example: const-wide vAA, #+BBBBBBBBBBBBBBBB
    class Instruction51l : public Instruction
    {
        /// @brief destination register (8 bits)
        std::uint8_t vAA;
        /// @brief second destination register (8 bits)
        std::uint8_t vBB;
        /// @brief wide value (64 bits)
        std::int64_t nBBBBBBBBBBBBBBBB;
    public:
        Instruction51l(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint8_t get_first_register() const
        {
            return vAA;
        }

        TYPES::Operand get_first_register_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        std::uint8_t get_second_register() const
        {
            return vBB;
        }

        TYPES::Operand get_second_register_type() const
        {
            return TYPES::Operand::REGISTER;
        }

        std::int64_t get_wide_value() const
        {
            return nBBBBBBBBBBBBBBBB;
        }

        TYPES::Operand get_wide_value_type() const
        {
            return TYPES::Operand::LITERAL;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op) + ", {v" + std::to_string(vAA) + 
                    ", v" + std::to_string(vBB) + "}, " + std::to_string(nBBBBBBBBBBBBBBBB);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + ", {v" + std::to_string(vAA) + 
                    ", v" + std::to_string(vBB) + "}, " + std::to_string(nBBBBBBBBBBBBBBBB);
        }
    };

    /// @brief Packed Switch instruction present in methods
    /// which make use of this kind of data
    class PackedSwitch : public Instruction
    {
        /// @brief number of targets
        std::uint16_t size;
        /// @brief first (and lowest) switch case value
        std::int32_t first_key;
        /// @brief targets where the program can jump
        std::vector<std::int32_t> targets;
    public:
        PackedSwitch(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint16_t get_number_of_targets() const
        {
            return size;
        }

        std::int32_t get_first_key() const
        {
            return first_key;
        }

        const std::vector<std::int32_t>& get_targets() const
        {
            return targets;
        }

        std::vector<std::int32_t>& get_targets()
        {
            return targets;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::stringstream output;
            
            output << DalvikOpcodes::get_instruction_name(op) + " (size)" + 
                std::to_string(size) + " (first/last key)" + std::to_string(first_key) + "[";

            for (const auto target : targets)
                output << "0x" << std::hex << target << ",";
            
            if (size > 0)
                output.seekp(-1, output.cur);
            
            output << "]";
            return output.str();
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " (size)" + 
                std::to_string(size) + " (first/last key)" + std::to_string(first_key) + "[";

            for (const auto target : targets)
                os << "0x" << std::hex << target << ",";
            
            if (size > 0)
                os.seekp(-1, os.cur);
            
            os << "]";
        }
    };

    /// @brief Sparse switch instruction present in methods
    /// which make use of this kind of data, this contain the
    /// keys
    class SparseSwitch : public Instruction
    {
        /// @brief Size of keys and targets
        std::uint16_t size;
        /// @brief keys checked and targets
        std::vector<std::tuple<std::int32_t, std::int32_t>> keys_targets;
    public:
        SparseSwitch(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);
        
        std::uint16_t get_size_of_targets() const
        {
            return size;
        }

        const std::vector<std::tuple<std::int32_t, std::int32_t>>& get_keys_targets() const
        {
            return keys_targets;
        }

        std::vector<std::tuple<std::int32_t, std::int32_t>>& get_keys_targets()
        {
            return keys_targets;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::stringstream output;
            
            output << DalvikOpcodes::get_instruction_name(op) << " (size)" << size << "[";

            for (const auto& key_target : keys_targets)
            {
                auto key = std::get<0>(key_target);
                auto target = std::get<1>(key_target);

                if (key < 0)
                    output << "-0x" << std::hex << key << ":";
                else
                    output << "0x" << std::hex << key << ":";

                if (target < 0)
                    output << "-0x" << std::hex << target << ":";
                else
                    output << "0x" << std::hex << target << ":";
                
                output << ",";
            }

            if (size > 0)
                output.seekp(-1, output.cur);

            output << "]";

            return output.str();
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << DalvikOpcodes::get_instruction_name(op) << " (size)" << size << "[";

            for (const auto& key_target : keys_targets)
            {
                auto key = std::get<0>(key_target);
                auto target = std::get<1>(key_target);

                if (key < 0)
                    os << "-0x" << std::hex << key << ":";
                else
                    os << "0x" << std::hex << key << ":";

                if (target < 0)
                    os << "-0x" << std::hex << target << ":";
                else
                    os << "0x" << std::hex << target << ":";
            }

            if (size > 0)
                os.seekp(-1, os.cur);

            os << "]";
        }
    };

    /// @brief Class present in methods which uses array data
    class FillArrayData : public Instruction
    {
        std::uint16_t element_width;
        std::uint32_t size;
        std::vector<std::uint8_t> data;
    public:
        FillArrayData(std::vector<uint8_t> &bytecode, std::size_t index, Parser * parser);

        std::uint16_t get_element_width() const
        {
            return element_width;
        }

        std::uint32_t get_size_of_data() const
        {
            return size;
        }

        const std::vector<std::uint8_t>& get_data() const
        {
            return data;
        }

        std::vector<std::uint8_t>& get_data()
        {
            return data;
        }

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::stringstream output;

            output << "(width)" << element_width << " (size)" << size << " [";

            for (auto byte : data)
            {
                output << "0x" << std::hex << static_cast<std::uint32_t>(byte) << ",";
            }

            if (size > 0)
                output.seekp(-1, output.cur);

            output << "]";

            return output.str();
        }        

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << "(width)" << element_width << " (size)" << size << " [";

            for (auto byte : data)
            {
                os << "0x" << std::hex << static_cast<std::uint32_t>(byte) << ",";
            }

            if (size > 0)
                os.seekp(-1, os.cur);

            os << "]";
        }
    };

    /// @brief In case there is an incorrect instruction
    /// this one holds all the opcodes and the length of
    /// previous instruction
    class DalvikIncorrectInstruction : public Instruction
    {
    public:
        DalvikIncorrectInstruction(std::vector<uint8_t> &bytecode, std::size_t index, std::uint32_t length)
            : Instruction(bytecode, index, dexinsttype_t::DEX_DALVIKINCORRECT, length)
        {
        }
        
        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            std::stringstream stream;

            stream << "DalvikInvalidInstruction [length: " << length << "][Opcodes: ";

            for (const auto val : op_codes)
                stream << std::hex << val << " ";
            
            stream.seekp(-1, stream.cur);

            stream << "]";

            return stream.str();
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream &os)
        {
            os << "DalvikInvalidInstruction [length: " << length << "][Opcodes: ";

            for (const auto val : op_codes)
                os << std::hex << val << " ";
            
            os.seekp(-1, os.cur);

            os << "]";
        }
    };
} // namespace DEX
} // namespace KUNAI

#endif