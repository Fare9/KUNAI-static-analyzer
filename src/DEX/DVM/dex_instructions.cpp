#include "dex_instructions.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * Instruction
         */
        Instruction::Instruction(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : dalvik_opcodes(dalvik_opcodes),
                                                                                                            length(0),
                                                                                                            OP(0)
        {
        }

        void Instruction::show_instruction()
        {
            std::cout << std::left << std::setfill(' ') << std::setw(25) << get_name() << get_output();
        }

        void Instruction::give_me_instruction(std::ostream &os)
        {
            os << std::left << std::setfill(' ') << std::setw(25) << get_name() << get_output();
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            return operands;
        }

        std::string Instruction::get_register_correct_representation(std::uint32_t reg)
        {
            std::string reg_representation;

            if (reg < (number_of_registers - number_of_parameters))
                reg_representation = "v" + std::to_string(reg);
            else
                reg_representation = "p" + std::to_string(reg - (number_of_registers - number_of_parameters));

            return reg_representation;
        }

        /**
         * Instruction00x
         */
        Instruction00x::Instruction00x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            this->set_length(0);
        }

        /**
         * Instruction10x
         */
        Instruction10x::Instruction10x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction10x");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Instruction10x high byte should be 0");

            this->set_OP(instruction[0]);
        }

        /**
         * Instruction12x
         */
        Instruction12x::Instruction12x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction12x");

            this->set_OP(instruction[0]);
            this->vA = (instruction[1] & 0x0F);
            this->vB = (instruction[1] & 0xF0) >> 4;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction12x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB}};

            return operands;
        }

        /**
         * Instruction11n
         */
        Instruction11n::Instruction11n(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction11n");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->nB = (instruction[1] & 0xF0) >> 4;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction11n::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::LITERAL, nB}};

            return operands;
        }

        /**
         * Instruction11x
         */
        Instruction11x::Instruction11x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction32x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction11x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA}};

            return operands;
        }

        /***
         * Instruction10t
         */
        Instruction10t::Instruction10t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction10t");

            if (instruction[1] == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction10t offset cannot be 0");

            this->set_OP(instruction[0]);
            this->nAA = static_cast<std::int8_t>(instruction[1]);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction10t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::OFFSET, nAA}};

            return operands;
        }

        /**
         * Instruction20t
         */
        Instruction20t::Instruction20t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction20t");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Error reading Instruction20t padding must be 0");

            this->set_OP(instruction[0]);
            this->nAAAA = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));

            if (this->nAAAA == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction20t offset cannot be 0");
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction20t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::OFFSET, nAAAA}};

            return operands;
        }

        /**
         * Instruction20bc
         */
        Instruction20bc::Instruction20bc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction20bc");

            this->set_OP(instruction[0]);
            this->nAA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction20bc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::LITERAL, nAA},
                {DVMTypes::Operand::LITERAL, nBBBB}};

            return operands;
        }

        /**
         * Instruction22x
         */
        Instruction22x::Instruction22x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBBBB}};

            return operands;
        }

        /**
         * Instruction21t
         */
        Instruction21t::Instruction21t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21t");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::int16_t *>(&instruction[2]));

            if (this->nBBBB == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction21t offset cannot be 0");
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::OFFSET, nBBBB}};

            return operands;
        }
        /**
         * Instruction21s
         */
        Instruction21s::Instruction21s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21s");

            this->set_OP(instruction[0]);
            this->vA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::int16_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21s::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::LITERAL, nBBBB}};

            return operands;
        }

        /**
         * Instruction21h
         */
        Instruction21h::Instruction21h(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];

            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21h");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBB_aux = *(reinterpret_cast<std::int16_t *>(&instruction[2]));

            switch (this->get_OP())
            {
            case 0x15:
                // const/high16 vAA, #+BBBB0000
                this->nBBBB = static_cast<std::int64_t>(this->nBBBB_aux) << 16;
                break;
            case 0x19:
                // const-wide/high16 vAA, #+BBBB000000000000
                this->nBBBB = static_cast<std::int64_t>(this->nBBBB_aux) << 48;
                break;
            default:
                this->nBBBB = static_cast<std::int64_t>(this->nBBBB_aux);
                break;
            }
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21h::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBB}};

            return operands;
        }

        /**
         * Instruction21c
         */
        Instruction21c::Instruction21c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21c");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->iBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        std::string Instruction21c::get_output()
        {
            std::string str = "";

            switch (this->get_kind())
            {
            case DVMTypes::Kind::STRING:
                str = this->get_dalvik_opcodes()->get_dalvik_string_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::TYPE:
                str = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::FIELD:
                str = this->get_dalvik_opcodes()->get_dalvik_static_field_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::METH:
                str = this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::PROTO:
                str = this->get_dalvik_opcodes()->get_dalvik_proto_by_id_str(iBBBB);
                break;
            default:
                str = std::to_string(iBBBB);
                break;
            }

            return this->get_register_correct_representation(vAA) + ", " + str;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::KIND, iBBBB}};

            return operands;
        }

        std::string *Instruction21c::get_source_str()
        {
            if (this->get_kind() == DVMTypes::Kind::STRING)
                return this->get_dalvik_opcodes()->get_dalvik_string_by_id(iBBBB);
            return nullptr;
        }
        Type *Instruction21c::get_source_typeid()
        {
            if (this->get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iBBBB);
            return nullptr;
        }
        FieldID *Instruction21c::get_source_static_field()
        {
            if (this->get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iBBBB);
            return nullptr;
        }
        MethodID *Instruction21c::get_source_method()
        {
            if (this->get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(iBBBB);
            return nullptr;
        }
        ProtoID *Instruction21c::get_source_proto()
        {
            if (this->get_kind() == DVMTypes::Kind::PROTO)
                return this->get_dalvik_opcodes()->get_dalvik_proto_by_id(iBBBB);
            return nullptr;
        }

        /**
         * Instruction23x
         */
        Instruction23x::Instruction23x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction23x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBB = instruction[2];
            this->vCC = instruction[3];
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction23x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBB},
                {DVMTypes::Operand::REGISTER, vCC},
            };

            return operands;
        }

        /**
         * Instruction22b
         */
        Instruction22b::Instruction22b(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22b");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBB = instruction[2];
            this->nCC = instruction[3];
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22b::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operators = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBB},
                {DVMTypes::Operand::LITERAL, nCC}};

            return operators;
        }

        /**
         * Instruction22t
         */
        Instruction22t::Instruction22t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22t");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0x0F;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->nCCCC = *(reinterpret_cast<std::int16_t *>(&instruction[2]));

            if (this->nCCCC == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction22t offset cannot be 0");
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::OFFSET, nCCCC}};

            return operands;
        }

        /**
         * Instruction22s
         */
        Instruction22s::Instruction22s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22s");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0x0F;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->nCCCC = *(reinterpret_cast<std::int16_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22s::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::LITERAL, nCCCC}};

            return operands;
        }

        /**
         * Instruction22c
         */
        Instruction22c::Instruction22c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22c");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->iCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        std::string Instruction22c::get_output()
        {
            std::string str;

            switch (this->get_kind())
            {
            case DVMTypes::Kind::TYPE:
                str = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(iCCCC);
                break;
            case DVMTypes::Kind::FIELD:
                str = this->get_dalvik_opcodes()->get_dalvik_static_field_by_id_str(iCCCC);
                break;
            default:
                str = std::to_string(iCCCC);
                break;
            }

            return this->get_register_correct_representation(vA) + ", " + this->get_register_correct_representation(vB) + ", " + str;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::KIND, iCCCC}};

            return operands;
        }

        Type *Instruction22c::get_third_operand_typeId()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iCCCC);
            return nullptr;
        }

        FieldID *Instruction22c::get_third_operand_FieldId()
        {
            if (get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iCCCC);
            return nullptr;
        }
        /**
         * Instruction22cs
         */
        Instruction22cs::Instruction22cs(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22c");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->iCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        std::string Instruction22cs::get_output()
        {
            std::string str;

            switch (this->get_kind())
            {
            case DVMTypes::Kind::TYPE:
                str = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(iCCCC);
                break;
            case DVMTypes::Kind::FIELD:
                str = this->get_dalvik_opcodes()->get_dalvik_static_field_by_id_str(iCCCC);
                break;
            default:
                str = std::to_string(iCCCC);
                break;
            }

            return this->get_register_correct_representation(vA) + ", " + this->get_register_correct_representation(vB) + ", " + str;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22cs::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::KIND, iCCCC}};

            return operands;
        }

        Type *Instruction22cs::get_third_operand_typeId()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iCCCC);
            return nullptr;
        }

        FieldID *Instruction22cs::get_third_operand_FieldId()
        {
            if (get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iCCCC);
            return nullptr;
        }
        /**
         * Instruction30t
         */
        Instruction30t::Instruction30t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction30t");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Error reading Instruction30t padding must be 0");

            this->set_OP(instruction[0]);
            this->nAAAAAAAA = *(reinterpret_cast<std::int32_t *>(&instruction[2]));

            if (this->nAAAAAAAA == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction30t offset cannot be 0");
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction30t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::OFFSET, nAAAAAAAA}};

            return operands;
        }

        /**
         * Instruction32x
         */
        Instruction32x::Instruction32x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];

            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction32x");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Instruction32x OP code high byte should be 0");

            this->set_OP(instruction[0]);
            this->vAAAA = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            this->vBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[4]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction32x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAAAA},
                {DVMTypes::Operand::REGISTER, vBBBB}};

            return operands;
        }

        /**
         * Instruction31i
         */
        Instruction31i::Instruction31i(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            this->set_length(6);
            std::uint8_t instruction[6];

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31i");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBB = *(reinterpret_cast<std::uint32_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction31i::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBBBBBB}};

            return operands;
        }

        /**
         * Instruction31t
         */
        Instruction31t::Instruction31t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31t");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBB = *(reinterpret_cast<std::int32_t *>(&instruction[2]));

            if (get_OP() == DVMTypes::OP_PACKED_SWITCH)
                this->type_of_switch = PACKED_SWITCH;
            else if (get_OP() == DVMTypes::OP_SPARSE_SWITCH)
                this->type_of_switch = SPARSE_SWITCH;
            else
                this->type_of_switch = NONE_SWITCH;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction31t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::OFFSET, nBBBBBBBB}};

            return operands;
        }

        /**
         * Instruction31c
         */
        Instruction31c::Instruction31c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31c");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->iBBBBBBBB = *(reinterpret_cast<std::uint32_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction31c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::KIND, iBBBBBBBB}};

            return operands;
        }

        /**
         * Instruction35c
         */
        Instruction35c::Instruction35c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t reg[5];

            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction35c");

            this->set_OP(instruction[0]);
            this->array_size = (instruction[1] & 0xf0) >> 4;
            this->type_index = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            reg[4] = (instruction[1] & 0x0f);
            reg[0] = (instruction[4] & 0x0f);
            reg[1] = (instruction[4] & 0xf0) >> 4;
            reg[2] = (instruction[5] & 0x0f);
            reg[3] = (instruction[5] & 0xf0) >> 4;

            if (this->array_size > 5)
                throw exceptions::InvalidInstruction("Error in array size of Instruction35c, cannot be greater than 5");

            for (size_t i = 0; i < this->array_size; i++)
                this->registers.push_back(reg[i]);
        }

        std::string Instruction35c::get_output()
        {
            std::string output = "";
            std::string ref = "";

            switch (get_kind())
            {
            case DVMTypes::Kind::TYPE:
                ref = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(type_index);
                break;
            case DVMTypes::Kind::METH:
                ref = this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(type_index);
                break;
            // don't know how to manage CALL_SITE Kind...
            default:
                ref = std::to_string(type_index);
            }

            for (size_t i = 0; i < array_size; i++)
                output += this->get_register_correct_representation(registers[i]) + ", ";

            output += ref;

            return output;
        }

        std::uint64_t Instruction35c::get_raw()
        {
            std::uint8_t raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            std::uint8_t reg[5] = {0, 0, 0, 0, 0};

            for (size_t i = 0; i < array_size; i++)
                reg[i] = registers[i];

            raw[0] = get_OP();
            raw[1] = array_size << 4 | reg[4];
            *reinterpret_cast<std::uint16_t *>(&raw[2]) = type_index;
            raw[4] = reg[1] << 4 | reg[0];
            raw[5] = reg[3] << 4 | reg[2];

            return *(reinterpret_cast<std::uint64_t *>(raw));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction35c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < array_size; i++)
                operands.push_back({DVMTypes::Operand::REGISTER, registers[i]});

            operands.push_back({DVMTypes::Operand::KIND, type_index});

            return operands;
        }

        std::uint8_t Instruction35c::get_operand_register(std::uint8_t index)
        {
            if (index >= array_size)
                return 0;
            return registers[index];
        }
        /**
         * Instruction3rc
         */
        Instruction3rc::Instruction3rc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            std::uint16_t vCCCC;
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction3rc");

            this->set_OP(instruction[0]);
            this->array_size = instruction[1];
            this->index = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            vCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[4]));

            for (std::uint16_t i = vCCCC; i < vCCCC + this->array_size; i++)
                registers.push_back(i);
        }

        std::string Instruction3rc::get_output()
        {
            std::string output = "";

            for (std::uint16_t i = 0; i < array_size; i++)
                output += this->get_register_correct_representation(registers[i]) + ", ";

            /**
             * This instruction is a little bit pain in the ass
             * there are different kind so we must decide here
             * which one we will use.
             *  	op {vCCCC .. vNNNN}, meth@BBBB
             *      op {vCCCC .. vNNNN}, site@BBBB
             *      op {vCCCC .. vNNNN}, type@BBBB
             *
             * Check for meth, and type, any other case, return
             * index as string.
             */
            if (this->get_kind() == DVMTypes::Kind::TYPE)
                output += this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(index);
            else if (this->get_kind() == DVMTypes::Kind::METH)
                output += this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(index);
            else
                output += std::to_string(index);

            return output;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction3rc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (std::uint16_t i = 0; i < array_size; i++)
                operands.push_back({DVMTypes::Operand::REGISTER, i});

            operands.push_back({DVMTypes::Operand::KIND, index});

            return operands;
        }

        std::any Instruction3rc::get_last_operand()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(index);
            else if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(index);
            else
                return index;
        }

        std::string Instruction3rc::get_last_operand_str()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(index);
            else if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(index);
            else
                return std::to_string(index);
        }

        Type *Instruction3rc::get_operands_type()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(index);
            else
                return nullptr;
        }

        std::string Instruction3rc::get_operands_type_str()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(index);
            else
                return "";
        }

        MethodID *Instruction3rc::get_operands_method()
        {
            if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(index);
            else
                return nullptr;
        }

        std::string Instruction3rc::get_operands_method_str()
        {
            if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(index);
            else
                return "";
        }

        std::uint8_t Instruction3rc::get_operand_register(std::uint8_t index)
        {
            if (index >= array_size)
                return 0;
            return registers[index];
        }
        /**
         * Instruction45cc
         */
        Instruction45cc::Instruction45cc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[8];
            std::uint8_t regC, regD, regE, regF, regG;
            this->set_length(8);

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction45cc");

            this->set_OP(instruction[0]);
            this->reg_count = (instruction[1] & 0xF0) >> 4;
            regG = instruction[1] & 0x0F;
            this->method_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            regD = (instruction[4] & 0xF0) >> 4;
            regC = instruction[4] & 0x0F;
            regF = (instruction[5] & 0xF0) >> 4;
            regE = instruction[5] & 0x0F;
            this->proto_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[6]));

            if (this->reg_count > 5)
                throw exceptions::InvalidInstruction("Error in reg_count from Instruction45cc cannot be greater than 5");

            if (this->reg_count > 0)
                this->registers.push_back(regC);
            if (this->reg_count > 1)
                this->registers.push_back(regD);
            if (this->reg_count > 2)
                this->registers.push_back(regE);
            if (this->reg_count > 3)
                this->registers.push_back(regF);
            if (this->reg_count > 4)
                this->registers.push_back(regG);
        }

        std::string Instruction45cc::get_output()
        {
            std::string output = "";
            std::string method = get_dalvik_opcodes()->get_dalvik_method_by_id_str(method_reference);
            std::string prototype = get_dalvik_opcodes()->get_dalvik_proto_by_id_str(proto_reference);

            for (size_t i = 0; i < reg_count; i++)
                output += this->get_register_correct_representation(registers[i]) + ", ";
            output += method + ", " + prototype;
            return output;
        }

        std::uint64_t Instruction45cc::get_raw()
        {
            std::uint8_t instruction[8];
            std::uint8_t regC = 0, regD = 0, regE = 0, regF = 0, regG = 0;

            if (reg_count > 0)
                regC = registers[0];
            if (reg_count > 1)
                regD = registers[1];
            if (reg_count > 2)
                regE = registers[2];
            if (reg_count > 3)
                regF = registers[3];
            if (reg_count > 4)
                regG = registers[4];

            instruction[0] = get_OP();
            instruction[1] = reg_count << 4 | regG;
            *(reinterpret_cast<std::uint16_t *>(&instruction[2])) = method_reference;
            instruction[4] = regD << 4 | regC;
            instruction[5] = regF << 4 | regE;
            *(reinterpret_cast<std::uint16_t *>(&instruction[6])) = proto_reference;
            return *(reinterpret_cast<std::uint64_t *>(&instruction[0]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction45cc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < reg_count; i++)
            {
                operands.push_back({DVMTypes::Operand::REGISTER, registers[i]});
            }
            operands.push_back({DVMTypes::Operand::KIND, method_reference});
            operands.push_back({DVMTypes::Operand::KIND, proto_reference});

            return operands;
        }

        std::uint8_t Instruction45cc::get_register(std::uint16_t index)
        {
            if (index >= reg_count)
                return 0;
            return registers[index];
        }

        /**
         * Instruction4rcc
         */
        Instruction4rcc::Instruction4rcc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[8];
            std::uint16_t vCCCC;
            this->set_length(8);

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction4rcc");

            this->set_OP(instruction[0]);
            this->reg_count = instruction[1];
            this->method_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            vCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[4]));
            this->proto_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[6]));

            for (size_t i = vCCCC; i < vCCCC + this->reg_count; i++)
                this->registers.push_back(i);
        }

        std::string Instruction4rcc::get_output()
        {
            std::string output = "";
            std::string method = get_dalvik_opcodes()->get_dalvik_method_by_id_str(method_reference);
            std::string prototype = get_dalvik_opcodes()->get_dalvik_proto_by_id_str(proto_reference);

            for (size_t i = 0; i < reg_count; i++)
                output = this->get_register_correct_representation(registers[i]) + ", ";
            output += method + ", " + prototype;
            return output;
        }

        std::uint64_t Instruction4rcc::get_raw()
        {
            std::uint8_t instruction[8];

            instruction[0] = get_OP();
            instruction[1] = reg_count;
            *(reinterpret_cast<std::uint16_t *>(&instruction[2])) = method_reference;
            *(reinterpret_cast<std::uint16_t *>(&instruction[4])) = registers[0];
            *(reinterpret_cast<std::uint16_t *>(&instruction[6])) = proto_reference;

            return *(reinterpret_cast<std::uint64_t *>(&instruction[0]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction4rcc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < reg_count; i++)
                operands.push_back({DVMTypes::Operand::REGISTER, registers[i]});
            operands.push_back({DVMTypes::Operand::KIND, method_reference});
            operands.push_back({DVMTypes::Operand::KIND, proto_reference});

            return operands;
        }

        std::uint16_t Instruction4rcc::get_register(std::uint16_t index)
        {
            if (index >= reg_count)
                return 0;
            return registers[index];
        }

        /**
         * Instruction51l
         */
        Instruction51l::Instruction51l(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[10];
            this->set_length(10);

            if (!KUNAI::read_data_file<std::uint8_t[10]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction51l");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::uint64_t *>(&instruction[2]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction51l::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBBBBBBBBBBBBBB}};

            return operands;
        }

        /**
         * PackedSwitch
         */
        PackedSwitch::PackedSwitch(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction_part1[8];
            std::int32_t aux;
            size_t buff_size;

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction_part1, 8, input_file))
                throw exceptions::DisassemblerException("Error disassembling PackedSwitch");

            this->ident = *(reinterpret_cast<std::uint16_t *>(&instruction_part1[0]));
            this->size = *(reinterpret_cast<std::uint16_t *>(&instruction_part1[2]));
            this->first_key = *(reinterpret_cast<std::int32_t *>(&instruction_part1[4]));

            this->set_OP(this->ident);

            buff_size = size;

            for (size_t i = 0; i < buff_size; i++)
            {
                if (!KUNAI::read_data_file<std::int32_t>(aux, sizeof(std::int32_t), input_file))
                    throw exceptions::DisassemblerException("Error disassembling PackedSwitch");
                this->targets.push_back(aux);
            }

            this->set_length(8 + this->targets.size() * 4);
        }

        PackedSwitch::~PackedSwitch()
        {
            if (!targets.empty())
                targets.clear();
        }

        std::string PackedSwitch::get_output()
        {
            std::stringstream str;

            str << "(size)" << size << " (first/last key)" << std::hex << first_key << " [";

            for (size_t i = 0; i < targets.size(); i++)
            {
                str << "0x" << std::hex << targets[i];
                if (i != (targets.size() - 1))
                    str << ",";
            }

            str << "]";

            return str.str();
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> PackedSwitch::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < targets.size(); i++)
                operands.push_back({DVMTypes::Operand::RAW, targets[i]});

            return operands;
        }

        /**
         * SparseSwitch
         */
        SparseSwitch::SparseSwitch(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction_part1[4];
            std::int32_t aux;
            std::vector<std::int32_t> keys;
            std::vector<std::int32_t> targets;

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction_part1, 4, input_file))
                throw exceptions::DisassemblerException("Error disassembling SparseSwitch");

            this->ident = *(reinterpret_cast<std::uint16_t *>(&instruction_part1[0]));
            this->size = *(reinterpret_cast<std::uint16_t *>(&instruction_part1[2]));

            this->set_OP(this->ident);

            for (size_t i = 0; i < this->size; i++)
            {
                if (!KUNAI::read_data_file<std::int32_t>(aux, sizeof(std::int32_t), input_file))
                    throw exceptions::DisassemblerException("Error disassembling SparseSwitch");

                keys.push_back(aux);
            }

            for (size_t i = 0; i < this->size; i++)
            {
                if (!KUNAI::read_data_file<std::int32_t>(aux, sizeof(std::int32_t), input_file))
                    throw exceptions::DisassemblerException("Error disassembling SparseSwitch");

                targets.push_back(aux);
            }

            for (size_t i = 0; i < this->size; i++)
                this->keys_targets.push_back({keys[i], targets[i]});

            this->set_length(4 + (sizeof(std::int32_t) * this->size) * 2);
        }

        SparseSwitch::~SparseSwitch()
        {
            if (!keys_targets.empty())
                keys_targets.clear();
        }

        std::string SparseSwitch::get_output()
        {
            std::stringstream str;

            str << "(size)" << size << " [";

            for (size_t i = 0; i < keys_targets.size(); i++)
            {
                if (std::get<0>(keys_targets[i]) < 0)
                    str << "-0x" << std::hex << std::get<0>(keys_targets[i]) << ":";
                else
                    str << "0x" << std::hex << std::get<0>(keys_targets[i]) << ":";
                if (std::get<1>(keys_targets[i]) < 0)
                    str << "-0x" << std::hex << std::get<1>(keys_targets[i]);
                else
                    str << "0x" << std::hex << std::get<1>(keys_targets[i]);

                if (i != (keys_targets.size() - 1))
                    str << ",";
            }

            str << "]";

            return str.str();
        }

        
        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> SparseSwitch::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < keys_targets.size(); i++)
            {
                operands.push_back({DVMTypes::Operand::LITERAL, std::get<0>(keys_targets[i])});
                operands.push_back({DVMTypes::Operand::OFFSET, std::get<1>(keys_targets[i])});
            }

            return operands;
        }

        std::int32_t SparseSwitch::get_target_by_key(std::int32_t key)
        {
            for (size_t i = 0; i < keys_targets.size(); i++)
            {
                if (std::get<0>(keys_targets[i]) == key)
                    return std::get<1>(keys_targets[i]);
            }

            return 0;
        }

        std::int32_t SparseSwitch::get_key_by_pos(size_t pos)
        {
            if (pos >= keys_targets.size())
                return 0;
            return std::get<0>(keys_targets[pos]);
        }

        std::int32_t SparseSwitch::get_target_by_pos(size_t pos)
        {
            if (pos >= keys_targets.size())
                return 0;
            return std::get<1>(keys_targets[pos]);
        }

        /**
         * FillArrayData
         */
        FillArrayData::FillArrayData(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction_part1[8];
            std::uint8_t aux;
            size_t buff_size;

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction_part1, 8, input_file))
                throw exceptions::DisassemblerException("Error disassembling FillArrayData");

            this->ident = *(reinterpret_cast<std::uint16_t *>(&instruction_part1[0]));
            this->element_width = *(reinterpret_cast<std::uint16_t *>(&instruction_part1[2]));
            this->size = *(reinterpret_cast<std::uint32_t *>(&instruction_part1[4]));

            this->set_OP(this->ident);

            buff_size = this->element_width * this->size;
            if (buff_size % 2 != 0)
                buff_size += 1;

            for (size_t i = 0; i < buff_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint8_t>(aux, 1, input_file))
                    throw exceptions::DisassemblerException("Error disassembling FillArrayData");
                data.push_back(aux);
            }

            this->set_length(8 + buff_size);
        }

        FillArrayData::~FillArrayData()
        {
            if (!data.empty())
                data.clear();
        }

        std::string FillArrayData::get_output()
        {
            std::stringstream str;

            str << "(width)" << element_width << " (size)" << size << " [";

            for (size_t i = 0; i < data.size(); i++)
            {
                str << "0x" << std::hex << static_cast<std::uint32_t>(data[i]);
                if (i != (data.size() - 1))
                    str << ",";
            }

            str << "]";

            return str.str();
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> FillArrayData::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < data.size(); i++)
                operands.push_back({DVMTypes::Operand::RAW, data[i]});

            return operands;
        }

        std::shared_ptr<Instruction> get_instruction_object(std::uint32_t opcode, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file)
        {
            std::shared_ptr<Instruction> instruction;

            switch (opcode)
            {
            case DVMTypes::Opcode::OP_NOP:
            {
                auto current_offset = input_file.tellg();
                char byte[2];

                input_file.read(byte, 2);

                input_file.seekg(current_offset); // in this case is necessary to set back the pointer

                if (byte[1] == 0x03)
                    instruction = std::make_shared<FillArrayData>(dalvik_opcodes, input_file); // filled-array-data
                else if (byte[1] == 0x01)
                    instruction = std::make_shared<PackedSwitch>(dalvik_opcodes, input_file); // packed-switch-data
                else if (byte[1] == 0x02)
                    instruction = std::make_shared<SparseSwitch>(dalvik_opcodes, input_file); // sparse-switch-data
                else
                    instruction = std::make_shared<Instruction10x>(dalvik_opcodes, input_file); // "nop"
                break;
            }
            case DVMTypes::Opcode::OP_MOVE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "move"
                break;
            case DVMTypes::Opcode::OP_MOVE_FROM16:
                instruction = std::make_shared<Instruction22x>(dalvik_opcodes, input_file); // "move/from16"
                break;
            case DVMTypes::Opcode::OP_MOVE_16:
                instruction = std::make_shared<Instruction32x>(dalvik_opcodes, input_file); // move/16
                break;
            case DVMTypes::Opcode::OP_MOVE_WIDE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // move-wide
                break;
            case DVMTypes::Opcode::OP_MOVE_WIDE_FROM16:
                instruction = std::make_shared<Instruction22x>(dalvik_opcodes, input_file); // move-wide/from16
                break;
            case DVMTypes::Opcode::OP_MOVE_WIDE_16:
                instruction = std::make_shared<Instruction32x>(dalvik_opcodes, input_file); // move-wide/16
                break;
            case DVMTypes::Opcode::OP_MOVE_OBJECT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // move-object
                break;
            case DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16:
                instruction = std::make_shared<Instruction22x>(dalvik_opcodes, input_file); // "move-object/from16"
                break;
            case DVMTypes::Opcode::OP_MOVE_OBJECT_16:
                instruction = std::make_shared<Instruction32x>(dalvik_opcodes, input_file); // "move-object/16"
                break;
            case DVMTypes::Opcode::OP_MOVE_RESULT:        // "move-result"
            case DVMTypes::Opcode::OP_MOVE_RESULT_WIDE:   // "move-result-wide"
            case DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT: // "move-result-object"
            case DVMTypes::Opcode::OP_MOVE_EXCEPTION:     // "move-exception"
            case DVMTypes::Opcode::OP_RETURN:             // "return"
            case DVMTypes::Opcode::OP_RETURN_WIDE:        // "return-wide"
            case DVMTypes::Opcode::OP_RETURN_OBJECT:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "return-object"
                break;
            case DVMTypes::Opcode::OP_RETURN_VOID:
                instruction = std::make_shared<Instruction10x>(dalvik_opcodes, input_file); // "return-void"
                break;
            case DVMTypes::Opcode::OP_CONST_4:
                instruction = std::make_shared<Instruction11n>(dalvik_opcodes, input_file); // "const/4"
                break;
            case DVMTypes::Opcode::OP_CONST_16:
                instruction = std::make_shared<Instruction21s>(dalvik_opcodes, input_file); // "const/16"
                break;
            case DVMTypes::Opcode::OP_CONST:
                instruction = std::make_shared<Instruction31i>(dalvik_opcodes, input_file); // "const"
                break;
            case DVMTypes::Opcode::OP_CONST_HIGH16:
                instruction = std::make_shared<Instruction21h>(dalvik_opcodes, input_file); // "const/high16"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE_16:
                instruction = std::make_shared<Instruction21s>(dalvik_opcodes, input_file); // "const-wide/16"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE_32:
                instruction = std::make_shared<Instruction31i>(dalvik_opcodes, input_file); // "const-wide/32"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE:
                instruction = std::make_shared<Instruction51l>(dalvik_opcodes, input_file); // "const-wide"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE_HIGH16:
                instruction = std::make_shared<Instruction21h>(dalvik_opcodes, input_file); // "const-wide/high16"
                break;
            case DVMTypes::Opcode::OP_CONST_STRING:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "const-string", Kind.STRING
                break;
            case DVMTypes::Opcode::OP_CONST_STRING_JUMBO:
                instruction = std::make_shared<Instruction31c>(dalvik_opcodes, input_file); // "const-string/jumbo", Kind.STRING
                break;
            case DVMTypes::Opcode::OP_CONST_CLASS:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "const-class", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_MONITOR_ENTER: // "monitor-enter"
            case DVMTypes::Opcode::OP_MONITOR_EXIT:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "monitor-exit"
                break;
            case DVMTypes::Opcode::OP_CHECK_CAST:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "check-cast", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_INSTANCE_OF:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "instance-of", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_ARRAY_LENGTH:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "array-length"
                break;
            case DVMTypes::Opcode::OP_NEW_INSTANCE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "new-instance", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_NEW_ARRAY:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "new-array", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_FILLED_NEW_ARRAY:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "filled-new-array", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_FILLED_NEW_ARRAY_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "filled-new-array/range", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_FILL_ARRAY_DATA:
                instruction = std::make_shared<Instruction31t>(dalvik_opcodes, input_file); // "fill-array-data"
                break;
            case DVMTypes::Opcode::OP_THROW:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "throw"
                break;
            case DVMTypes::Opcode::OP_GOTO:
                instruction = std::make_shared<Instruction10t>(dalvik_opcodes, input_file); // "goto"
                break;
            case DVMTypes::Opcode::OP_GOTO_16:
                instruction = std::make_shared<Instruction20t>(dalvik_opcodes, input_file); // "goto/16"
                break;
            case DVMTypes::Opcode::OP_GOTO_32:
                instruction = std::make_shared<Instruction30t>(dalvik_opcodes, input_file); // "goto/32"
                break;
            case DVMTypes::Opcode::OP_PACKED_SWITCH: // "packed-switch"
            case DVMTypes::Opcode::OP_SPARSE_SWITCH:
                instruction = std::make_shared<Instruction31t>(dalvik_opcodes, input_file); // "sparse-switch"
                break;
            case DVMTypes::Opcode::OP_CMPL_FLOAT:  // "cmpl-float"
            case DVMTypes::Opcode::OP_CMPG_FLOAT:  // "cmpg-float"
            case DVMTypes::Opcode::OP_CMPL_DOUBLE: // "cmpl-double"
            case DVMTypes::Opcode::OP_CMPG_DOUBLE: // "cmpg-double"
            case DVMTypes::Opcode::OP_CMP_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "cmp-long"
                break;
            case DVMTypes::Opcode::OP_IF_EQ: // "if-eq"
            case DVMTypes::Opcode::OP_IF_NE: // "if-ne"
            case DVMTypes::Opcode::OP_IF_LT: // "if-lt"
            case DVMTypes::Opcode::OP_IF_GE: // "if-ge"
            case DVMTypes::Opcode::OP_IF_GT: // "if-gt"
            case DVMTypes::Opcode::OP_IF_LE:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-le"
                break;
            case DVMTypes::Opcode::OP_IF_EQZ: // "if-eqz"
            case DVMTypes::Opcode::OP_IF_NEZ: // "if-nez"
            case DVMTypes::Opcode::OP_IF_LTZ: // "if-ltz"
            case DVMTypes::Opcode::OP_IF_GEZ: // "if-gez"
            case DVMTypes::Opcode::OP_IF_GTZ: // "if-gtz"
            case DVMTypes::Opcode::OP_IF_LEZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-lez"
                break;
            case DVMTypes::Opcode::OP_UNUSED_3E: // "unused"
            case DVMTypes::Opcode::OP_UNUSED_3F: // "unused"
            case DVMTypes::Opcode::OP_UNUSED_40: // "unused"
            case DVMTypes::Opcode::OP_UNUSED_41: // "unused"
            case DVMTypes::Opcode::OP_UNUSED_42: // "unused"
            case DVMTypes::Opcode::OP_UNUSED_43:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_AGET:         // "aget"
            case DVMTypes::Opcode::OP_AGET_WIDE:    // "aget-wide"
            case DVMTypes::Opcode::OP_AGET_OBJECT:  // "aget-object"
            case DVMTypes::Opcode::OP_AGET_BOOLEAN: // "aget-boolean"
            case DVMTypes::Opcode::OP_AGET_BYTE:    // "aget-byte"
            case DVMTypes::Opcode::OP_AGET_CHAR:    // "aget-char"
            case DVMTypes::Opcode::OP_AGET_SHORT:   // "aget-short"
            case DVMTypes::Opcode::OP_APUT:         // "aput"
            case DVMTypes::Opcode::OP_APUT_WIDE:    // "aput-wide"
            case DVMTypes::Opcode::OP_APUT_OBJECT:  // "aput-object"
            case DVMTypes::Opcode::OP_APUT_BOOLEAN: // "aput-boolean"
            case DVMTypes::Opcode::OP_APUT_BYTE:    // "aput-byte"
            case DVMTypes::Opcode::OP_APUT_CHAR:    // "aput-char"
            case DVMTypes::Opcode::OP_APUT_SHORT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-short"
                break;
            case DVMTypes::Opcode::OP_IGET:         // "iget"
            case DVMTypes::Opcode::OP_IGET_WIDE:    // "iget-wide"
            case DVMTypes::Opcode::OP_IGET_OBJECT:  // "iget-object"
            case DVMTypes::Opcode::OP_IGET_BOOLEAN: // "iget-boolean"
            case DVMTypes::Opcode::OP_IGET_BYTE:    // "iget-byte"
            case DVMTypes::Opcode::OP_IGET_CHAR:    // "iget-char"
            case DVMTypes::Opcode::OP_IGET_SHORT:   // "iget-short"
            case DVMTypes::Opcode::OP_IPUT:         // "iput"
            case DVMTypes::Opcode::OP_IPUT_WIDE:    // "iput-wide"
            case DVMTypes::Opcode::OP_IPUT_OBJECT:  // "iput-object"
            case DVMTypes::Opcode::OP_IPUT_BOOLEAN: // "iput-boolean"
            case DVMTypes::Opcode::OP_IPUT_BYTE:    // "iput-byte"
            case DVMTypes::Opcode::OP_IPUT_CHAR:    // "iput-char"
            case DVMTypes::Opcode::OP_IPUT_SHORT:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-short"
                break;
            case DVMTypes::Opcode::OP_SGET:         // "sget"
            case DVMTypes::Opcode::OP_SGET_WIDE:    // "sget-wide"
            case DVMTypes::Opcode::OP_SGET_OBJECT:  // "sget-object"
            case DVMTypes::Opcode::OP_SGET_BOOLEAN: // "sget-boolean"
            case DVMTypes::Opcode::OP_SGET_BYTE:    // "sget-byte"
            case DVMTypes::Opcode::OP_SGET_CHAR:    // "sget-char"
            case DVMTypes::Opcode::OP_SGET_SHORT:   // "sget-short"
            case DVMTypes::Opcode::OP_SPUT:         // "sput"
            case DVMTypes::Opcode::OP_SPUT_WIDE:    // "sput-wide"
            case DVMTypes::Opcode::OP_SPUT_OBJECT:  // "sput-object"
            case DVMTypes::Opcode::OP_SPUT_BOOLEAN: // "sput-boolean"
            case DVMTypes::Opcode::OP_SPUT_BYTE:    // "sput-byte"
            case DVMTypes::Opcode::OP_SPUT_CHAR:    // "sput-char"
            case DVMTypes::Opcode::OP_SPUT_SHORT:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-short"
                break;
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL: // "invoke-virtual"
            case DVMTypes::Opcode::OP_INVOKE_SUPER:   // "invoke-super"
            case DVMTypes::Opcode::OP_INVOKE_DIRECT:  // "invoke-direct"
            case DVMTypes::Opcode::OP_INVOKE_STATIC:  // "invoke-static"
            case DVMTypes::Opcode::OP_INVOKE_INTERFACE:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-interface"
                break;
            case DVMTypes::Opcode::OP_UNUSED_73:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE: // "invoke-virtual/range"
            case DVMTypes::Opcode::OP_INVOKE_SUPER_RANGE:   // "invoke-super/range"
            case DVMTypes::Opcode::OP_INVOKE_DIRECT_RANGE:  // "invoke-direct/range"
            case DVMTypes::Opcode::OP_INVOKE_STATIC_RANGE:  // "invoke-static/range"
            case DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-interface/range"
                break;
            case DVMTypes::Opcode::OP_UNUSED_79: // "unused"
            case DVMTypes::Opcode::OP_UNUSED_7A:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_NEG_INT:         // "neg-int"
            case DVMTypes::Opcode::OP_NOT_INT:         // "not-int"
            case DVMTypes::Opcode::OP_NEG_LONG:        // "neg-long"
            case DVMTypes::Opcode::OP_NOT_LONG:        // "not-long"
            case DVMTypes::Opcode::OP_NEG_FLOAT:       // "neg-float"
            case DVMTypes::Opcode::OP_NEG_DOUBLE:      // "neg-double"
            case DVMTypes::Opcode::OP_INT_TO_LONG:     // "int-to-long"
            case DVMTypes::Opcode::OP_INT_TO_FLOAT:    // "int-to-float"
            case DVMTypes::Opcode::OP_INT_TO_DOUBLE:   // "int-to-double"
            case DVMTypes::Opcode::OP_LONG_TO_INT:     // "long-to-int"
            case DVMTypes::Opcode::OP_LONG_TO_FLOAT:   // "long-to-float"
            case DVMTypes::Opcode::OP_LONG_TO_DOUBLE:  // "long-to-double"
            case DVMTypes::Opcode::OP_FLOAT_TO_INT:    // "float-to-int"
            case DVMTypes::Opcode::OP_FLOAT_TO_LONG:   // "float-to-long"
            case DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE: // "float-to-double"
            case DVMTypes::Opcode::OP_DOUBLE_TO_INT:   // "double-to-int"
            case DVMTypes::Opcode::OP_DOUBLE_TO_LONG:  // "double-to-long"
            case DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT: // "double-to-float"
            case DVMTypes::Opcode::OP_INT_TO_BYTE:     // "int-to-byte"
            case DVMTypes::Opcode::OP_INT_TO_CHAR:     // "int-to-char"
            case DVMTypes::Opcode::OP_INT_TO_SHORT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-short"
                break;
            case DVMTypes::Opcode::OP_ADD_INT:    // "add-int"
            case DVMTypes::Opcode::OP_SUB_INT:    // "sub-int"
            case DVMTypes::Opcode::OP_MUL_INT:    // "mul-int"
            case DVMTypes::Opcode::OP_DIV_INT:    // "div-int"
            case DVMTypes::Opcode::OP_REM_INT:    // "rem-int"
            case DVMTypes::Opcode::OP_AND_INT:    // "and-int"
            case DVMTypes::Opcode::OP_OR_INT:     // "or-int"
            case DVMTypes::Opcode::OP_XOR_INT:    // "xor-int"
            case DVMTypes::Opcode::OP_SHL_INT:    // "shl-int"
            case DVMTypes::Opcode::OP_SHR_INT:    // "shr-int"
            case DVMTypes::Opcode::OP_USHR_INT:   // "ushr-int"
            case DVMTypes::Opcode::OP_ADD_LONG:   // "add-long"
            case DVMTypes::Opcode::OP_SUB_LONG:   // "sub-long"
            case DVMTypes::Opcode::OP_MUL_LONG:   // "mul-long"
            case DVMTypes::Opcode::OP_DIV_LONG:   // "div-long"
            case DVMTypes::Opcode::OP_REM_LONG:   // "rem-long"
            case DVMTypes::Opcode::OP_AND_LONG:   // "and-long"
            case DVMTypes::Opcode::OP_OR_LONG:    // "or-long"
            case DVMTypes::Opcode::OP_XOR_LONG:   // "xor-long"
            case DVMTypes::Opcode::OP_SHL_LONG:   // "shl-long"
            case DVMTypes::Opcode::OP_SHR_LONG:   // "shr-long"
            case DVMTypes::Opcode::OP_USHR_LONG:  // "ushr-long"
            case DVMTypes::Opcode::OP_ADD_FLOAT:  // "add-float"
            case DVMTypes::Opcode::OP_SUB_FLOAT:  // "sub-float"
            case DVMTypes::Opcode::OP_MUL_FLOAT:  // "mul-float"
            case DVMTypes::Opcode::OP_DIV_FLOAT:  // "div-float"
            case DVMTypes::Opcode::OP_REM_FLOAT:  // "rem-float"
            case DVMTypes::Opcode::OP_ADD_DOUBLE: // "add-double"
            case DVMTypes::Opcode::OP_SUB_DOUBLE: // "sub-double"
            case DVMTypes::Opcode::OP_MUL_DOUBLE: // "mul-double"
            case DVMTypes::Opcode::OP_DIV_DOUBLE: // "div-double"
            case DVMTypes::Opcode::OP_REM_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "rem-double"
                break;
            case DVMTypes::Opcode::OP_ADD_INT_2ADDR:    // "add-int/2addr"
            case DVMTypes::Opcode::OP_SUB_INT_2ADDR:    // "sub-int/2addr"
            case DVMTypes::Opcode::OP_MUL_INT_2ADDR:    // "mul-int/2addr"
            case DVMTypes::Opcode::OP_DIV_INT_2ADDR:    // "div-int/2addr"
            case DVMTypes::Opcode::OP_REM_INT_2ADDR:    // "rem-int/2addr"
            case DVMTypes::Opcode::OP_AND_INT_2ADDR:    // "and-int/2addr"
            case DVMTypes::Opcode::OP_OR_INT_2ADDR:     // "or-int/2addr"
            case DVMTypes::Opcode::OP_XOR_INT_2ADDR:    // "xor-int/2addr"
            case DVMTypes::Opcode::OP_SHL_INT_2ADDR:    // "shl-int/2addr"
            case DVMTypes::Opcode::OP_SHR_INT_2ADDR:    // "shr-int/2addr"
            case DVMTypes::Opcode::OP_USHR_INT_2ADDR:   // "ushr-int/2addr"
            case DVMTypes::Opcode::OP_ADD_LONG_2ADDR:   // "add-long/2addr"
            case DVMTypes::Opcode::OP_SUB_LONG_2ADDR:   // "sub-long/2addr"
            case DVMTypes::Opcode::OP_MUL_LONG_2ADDR:   // "mul-long/2addr"
            case DVMTypes::Opcode::OP_DIV_LONG_2ADDR:   // "div-long/2addr"
            case DVMTypes::Opcode::OP_REM_LONG_2ADDR:   // "rem-long/2addr"
            case DVMTypes::Opcode::OP_AND_LONG_2ADDR:   // "and-long/2addr"
            case DVMTypes::Opcode::OP_OR_LONG_2ADDR:    // "or-long/2addr"
            case DVMTypes::Opcode::OP_XOR_LONG_2ADDR:   // "xor-long/2addr"
            case DVMTypes::Opcode::OP_SHL_LONG_2ADDR:   // "shl-long/2addr"
            case DVMTypes::Opcode::OP_SHR_LONG_2ADDR:   // "shr-long/2addr"
            case DVMTypes::Opcode::OP_USHR_LONG_2ADDR:  // "ushr-long/2addr"
            case DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR:  // "add-float/2addr"
            case DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR:  // "sub-float/2addr"
            case DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR:  // "mul-float/2addr"
            case DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR:  // "div-float/2addr"
            case DVMTypes::Opcode::OP_REM_FLOAT_2ADDR:  // "rem-float/2addr"
            case DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR: // "add-double/2addr"
            case DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR: // "sub-double/2addr"
            case DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR: // "mul-double/2addr"
            case DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR: // "div-double/2addr"
            case DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "rem-double/2addr"
                break;
            case DVMTypes::Opcode::OP_ADD_INT_LIT16: // "add-int/lit16"
            case DVMTypes::Opcode::OP_RSUB_INT:      // "rsub-int"
            case DVMTypes::Opcode::OP_MUL_INT_LIT16: // "mul-int/lit16"
            case DVMTypes::Opcode::OP_DIV_INT_LIT16: // "div-int/lit16"
            case DVMTypes::Opcode::OP_REM_INT_LIT16: // "rem-int/lit16"
            case DVMTypes::Opcode::OP_AND_INT_LIT16: // "and-int/lit16"
            case DVMTypes::Opcode::OP_OR_INT_LIT16:  // "or-int/lit16"
            case DVMTypes::Opcode::OP_XOR_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "xor-int/lit16"
                break;
            case DVMTypes::Opcode::OP_ADD_INT_LIT8:  // "add-int/lit8"
            case DVMTypes::Opcode::OP_RSUB_INT_LIT8: // "rsub-int/lit8"
            case DVMTypes::Opcode::OP_MUL_INT_LIT8:  // "mul-int/lit8"
            case DVMTypes::Opcode::OP_DIV_INT_LIT8:  // "div-int/lit8"
            case DVMTypes::Opcode::OP_REM_INT_LIT8:  // "rem-int/lit8"
            case DVMTypes::Opcode::OP_AND_INT_LIT8:  // "and-int/lit8"
            case DVMTypes::Opcode::OP_OR_INT_LIT8:   // "or-int/lit8"
            case DVMTypes::Opcode::OP_XOR_INT_LIT8:  // "xor-int/lit8"
            case DVMTypes::Opcode::OP_SHL_INT_LIT8:  // "shl-int/lit8"
            case DVMTypes::Opcode::OP_SHR_INT_LIT8:  // "shr-int/lit8"
            case DVMTypes::Opcode::OP_USHR_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "ushr-int/lit8"
                break;
            case DVMTypes::Opcode::OP_IGET_VOLATILE:            // "unused"
            case DVMTypes::Opcode::OP_IPUT_VOLATILE:            // "unused"
            case DVMTypes::Opcode::OP_SGET_VOLATILE:            // "unused"
            case DVMTypes::Opcode::OP_SPUT_VOLATILE:            // "unused"
            case DVMTypes::Opcode::OP_IGET_OBJECT_VOLATILE:     // "unused"
            case DVMTypes::Opcode::OP_IGET_WIDE_VOLATILE:       // "unused"
            case DVMTypes::Opcode::OP_IPUT_WIDE_VOLATILE:       // "unused"
            case DVMTypes::Opcode::OP_SGET_WIDE_VOLATILE:       // "unused"
            case DVMTypes::Opcode::OP_SPUT_WIDE_VOLATILE:       // "unused"
            case DVMTypes::Opcode::OP_BREAKPOINT:               // "unused"
            case DVMTypes::Opcode::OP_THROW_VERIFICATION_ERROR: // "unused"
            case DVMTypes::Opcode::OP_EXECUTE_INLINE:           // "unused"
            case DVMTypes::Opcode::OP_EXECUTE_INLINE_RANGE:     // "unused"
            case DVMTypes::Opcode::OP_INVOKE_OBJECT_INIT_RANGE: // "unused"
            case DVMTypes::Opcode::OP_RETURN_VOID_BARRIER:      // "unused"
            case DVMTypes::Opcode::OP_IGET_QUICK:               // "unused"
            case DVMTypes::Opcode::OP_IGET_WIDE_QUICK:          // "unused"
            case DVMTypes::Opcode::OP_IGET_OBJECT_QUICK:        // "unused"
            case DVMTypes::Opcode::OP_IPUT_QUICK:               // "unused"
            case DVMTypes::Opcode::OP_IPUT_WIDE_QUICK:          // "unused"
            case DVMTypes::Opcode::OP_IPUT_OBJECT_QUICK:        // "unused"
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL_QUICK:     // "unused"
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL_QUICK_RANGE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK:
                instruction = std::make_shared<Instruction45cc>(dalvik_opcodes, input_file); // "invoke-polymorphic" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK_RANGE:
                instruction = std::make_shared<Instruction4rcc>(dalvik_opcodes, input_file); // "invoke-polymorphic/range" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_IPUT_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-custom" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_SGET_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-custom/range" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_SPUT_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "const-method-handle" # Dalvik 039
                break;
            case DVMTypes::Opcode::OP_CONST_METHOD_TYPE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // 'const-method-type' # Dalvik 039
                break;
            default:
                std::string msg = "Invalud Instruction '" + std::to_string(opcode) + "'";
                throw exceptions::InvalidInstruction(msg);
            }

            return instruction;
        }

        template <typename Base, typename T>
        inline bool instanceof (const T *)
        {
            return std::is_base_of<Base, T>::value;
        }

        /**
         * @brief Determine the next offsets inside the bytecode of an :class:`EncodedMethod`.
         *        The offsets are calculated in number of bytes from the start of the method.
         *        Note, that offsets inside the bytecode are denoted in 16bit units but this method returns actual bytes!
         *
         *        Offsets inside the opcode are counted from the beginning of the opcode.
         *
         *        The returned type is a list, as branching opcodes will have multiple paths.
         *        `if` and `switch` opcodes will return more than one item in the list, while
         *        `throw`, `return` and `goto` opcodes will always return a list with length one.
         *
         *        An offset of -1 indicates that the method is exited, for example by `throw` or `return`.
         *
         *        If the entered opcode is not branching or jumping, an empty list is returned.
         *
         * @param instr: instruction to calculate the next one in case this is a Goto, if, switch.
         * @param curr_idx: current idx to calculate new idx.
         * @param instructions: all the instructions from the method.
         *
         */
        std::vector<std::int64_t> determine_next(std::shared_ptr<Instruction> instr,
                                                 std::uint64_t curr_idx,
                                                 std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions)
        {
            auto op_value = instr->get_OP();

            if ((op_value == DVMTypes::Opcode::OP_THROW) ||
                ((op_value >= DVMTypes::Opcode::OP_RETURN_VOID) &&
                 (op_value <= DVMTypes::Opcode::OP_RETURN_OBJECT)))
            {
                return {-1};
            }
            else if ((op_value >= DVMTypes::Opcode::OP_GOTO) &&
                     (op_value <= DVMTypes::Opcode::OP_GOTO_32))
            {
                std::int32_t offset;
                switch (op_value)
                {
                case DVMTypes::Opcode::OP_GOTO:
                {
                    auto goto_instr = std::dynamic_pointer_cast<Instruction10t>(instr);
                    offset = goto_instr->get_offset() * 2;
                    break;
                }
                case DVMTypes::Opcode::OP_GOTO_16:
                {
                    auto goto_instr = std::dynamic_pointer_cast<Instruction20t>(instr);
                    offset = goto_instr->get_offset() * 2;
                    break;
                }
                case DVMTypes::Opcode::OP_GOTO_32:
                {
                    auto goto_instr = std::dynamic_pointer_cast<Instruction30t>(instr);
                    offset = goto_instr->get_offset() * 2;
                    break;
                }
                }

                return {offset + static_cast<std::int64_t>(curr_idx)};
            }
            else if ((op_value >= DVMTypes::Opcode::OP_IF_EQ) &&
                     (op_value <= DVMTypes::Opcode::OP_IF_LEZ))
            {
                std::int32_t offset;

                if ((op_value >= DVMTypes::Opcode::OP_IF_EQ) &&
                    (op_value <= DVMTypes::Opcode::OP_IF_LE))
                {
                    auto if_instr = std::dynamic_pointer_cast<Instruction22t>(instr);
                    offset = if_instr->get_ref() * 2;
                }
                else if ((op_value >= DVMTypes::Opcode::OP_IF_EQZ) &&
                         (op_value <= DVMTypes::Opcode::OP_IF_LEZ))
                {
                    auto if_instr = std::dynamic_pointer_cast<Instruction21t>(instr);
                    offset = if_instr->get_ref() * 2;
                }

                return {static_cast<std::int64_t>(curr_idx) + instr->get_length(), offset + static_cast<std::int64_t>(curr_idx)};
            }
            else if ((op_value == DVMTypes::Opcode::OP_PACKED_SWITCH) ||
                     (op_value == DVMTypes::Opcode::OP_SPARSE_SWITCH))
            {
                std::vector<std::int64_t> x = {static_cast<std::int64_t>(curr_idx) + instr->get_length()};

                auto switch_instr = std::dynamic_pointer_cast<Instruction31t>(instr);

                switch (switch_instr->get_type_of_switch())
                {
                case Instruction31t::PACKED_SWITCH:
                {
                    auto packed_switch = switch_instr->get_packed_switch();
                    auto targets = packed_switch->get_targets();

                    for (auto target : targets)
                        x.push_back(curr_idx + target * 2);
                }
                break;
                case Instruction31t::SPARSE_SWITCH:
                {
                    auto sparse_switch = switch_instr->get_sparse_switch();
                    auto targets = sparse_switch->get_keys_targets();

                    for (auto target : targets)
                        x.push_back(curr_idx + std::get<1>(target) * 2);
                }
                break;
                default:
                break;
                }

                return x;
            }

            return {};
        }

        std::vector<exceptions_data> determine_exception(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::shared_ptr<EncodedMethod> method)
        {
            if (method->get_code_item()->get_number_of_try_items() <= 0)
                return {};

            std::map<std::uint64_t, std::vector<std::vector<std::any>>> h_off;

            for (size_t i = 0; i < method->get_code_item()->get_number_of_try_items(); i++)
            {
                auto try_item = method->get_code_item()->get_try_item_by_pos(i);

                auto offset_handler = try_item->get_handler_off() +
                                      method->get_code_item()->get_encoded_catch_handler_offset();

                h_off[offset_handler].push_back({try_item});
            }

            for (size_t i = 0; i < method->get_code_item()->get_encoded_catch_handler_list_size(); i++)
            {
                auto encoded_catch_handler = method->get_code_item()->get_encoded_catch_handler_by_pos(i);

                if (h_off.find(encoded_catch_handler->get_offset()) == h_off.end())
                    continue;

                for (size_t j = 0; j < h_off[encoded_catch_handler->get_offset()].size(); j++)
                {
                    h_off[encoded_catch_handler->get_offset()][j].push_back(encoded_catch_handler);
                }
            }

            std::vector<exceptions_data> exceptions;

            for (auto it_map = h_off.begin(); it_map != h_off.end(); it_map++)
            {
                for (size_t i = 0; i < it_map->second.size(); i++)
                {
                    auto value = it_map->second[i];

                    auto try_value = std::any_cast<std::shared_ptr<KUNAI::DEX::TryItem>>(value[0]);
                    auto handler_catch = std::any_cast<std::shared_ptr<KUNAI::DEX::EncodedCatchHandler>>(value[1]);

                    exceptions_data z;

                    z.try_value_start_addr = try_value->get_start_addr() * 2;
                    z.try_value_end_addr = (try_value->get_start_addr() * 2) + (try_value->get_insn_count() * 2) - 1;

                    for (size_t j = 0; j < handler_catch->get_size_of_handlers(); j++)
                    {
                        auto catch_type_pair = handler_catch->get_handler_by_pos(j);
                        std::string cm_type;

                        if (catch_type_pair->get_exception_type()->get_type() == Type::FUNDAMENTAL)
                            cm_type = reinterpret_cast<Fundamental *>(catch_type_pair->get_exception_type())->print_fundamental_type();
                        else if (catch_type_pair->get_exception_type()->get_type() == Type::CLASS)
                            cm_type = reinterpret_cast<Class *>(catch_type_pair->get_exception_type())->get_name();
                        else
                            cm_type = catch_type_pair->get_exception_type()->get_raw();

                        z.handler.push_back({cm_type, catch_type_pair->get_exception_handler_addr() * 2});
                    }

                    if (handler_catch->get_size_of_handlers() <= 0)
                        z.handler.push_back({"Ljava/lang/Throwable;", handler_catch->get_catch_all_addr() * 2});

                    exceptions.push_back(z);
                }
            }

            return exceptions;
        }

    }
}