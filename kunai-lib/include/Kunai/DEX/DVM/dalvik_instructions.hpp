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
#include "Kunai/DEX/DVM/dvm_types.hpp"
#include "Kunai/DEX/DVM/dalvik_opcodes.hpp"

#include <iostream>
#include <span>
#include <string>

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

    public:

        /// @brief Constructor of the Instruction, here is applied
        /// the parsing of the opcodes
        /// @param bytecode 
        /// @param index 
        /// @param instruction_type 
        Instruction(std::vector<uint8_t>& bytecode, std::size_t index, dexinsttype_t instruction_type)
            : instruction_type(instruction_type), length(0), op(0), op_codes({})
        {
        }

        Instruction(std::vector<uint8_t>& bytecode, std::size_t index, dexinsttype_t instruction_type, std::uint32_t length)
            : instruction_type(instruction_type), length(length), op(0)
        {
            /// op_codes we can read here for all the classes
            /// that derives from Instruction, we have that is
            /// the bytecode from the index to index+length
            op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};
        }

        /// @brief Destructor of the instruction
        virtual ~Instruction();

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

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return "";
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream& os)
        {
            os << "";
        }

        /// @brief Return the op codes in raw from the instruction
        /// @return constant reference with op codes in raw
        virtual const std::span<std::uint8_t>& get_opcodes()
        {
            return op_codes;
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
        Instruction00x(std::vector<uint8_t>& bytecode, std::size_t index) :
            Instruction(bytecode, index, dexinsttype_t::DEX_INSTRUCTION00X)
        {
        }
    };

    /// @brief Instruction for wasting cycles. It represents
    /// a nop, it has a length of 2
    class Instruction10x : public Instruction
    {
    public:
        Instruction10x(std::vector<uint8_t>& bytecode, std::size_t index);

        /// @brief Return a string with the representation of the instruction
        /// @return string with instruction
        virtual std::string print_instruction()
        {
            return DalvikOpcodes::get_instruction_name(op);
        }

        /// @brief Print the instruction on a given stream
        /// @param os stream where to print the instruction
        virtual void print_instruction(std::ostream& os)
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
        Instruction12x(std::vector<uint8_t>& bytecode, std::size_t index);

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
        virtual void print_instruction(std::ostream& os)
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
        Instruction11n(std::vector<uint8_t>& bytecode, std::size_t index);

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
        virtual void print_instruction(std::ostream& os)
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
        Instruction11x(std::vector<uint8_t>& bytecode, std::size_t index);

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
        virtual void print_instruction(std::ostream& os)
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
        Instruction10t(std::vector<uint8_t>& bytecode, std::size_t index);
        
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
        virtual void print_instruction(std::ostream& os)
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
        Instruction20t(std::vector<uint8_t>& bytecode, std::size_t index);

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
        virtual void print_instruction(std::ostream& os)
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
        Instruction20bc(std::vector<uint8_t>& bytecode, std::size_t index);

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
        virtual void print_instruction(std::ostream& os)
        {
            os << DalvikOpcodes::get_instruction_name(op) + " " +
                    std::to_string(nAA) + ", kind@" + std::to_string(nBBBB);
        }
    };
} // namespace DEX
} // namespace KUNAI


#endif